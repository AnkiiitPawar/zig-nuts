//! The No-U-Turn sampler as Stan implements it (stan/mcmc/hmc/nuts/base_nuts.hpp,
//! Stan 2.26 and later): multinomial sampling of the trajectory, biased
//! progressive sampling between subtrees, the generalised no-U-turn criterion
//! on sharp momenta with the extra checks across subtree boundaries, and a
//! diagonal Euclidean metric. Warmup follows Stan's defaults: dual-averaging
//! step size adaptation toward an acceptance statistic of 0.8, and a diagonal
//! inverse metric estimated in doubling windows of 25, 50, 100, ... draws
//! between a 75-draw initial buffer and a 50-draw terminal buffer.

const std = @import("std");
const smath = @import("math.zig");
const models = @import("models.zig");

pub const Config = struct {
    num_warmup: u32 = 1000,
    num_samples: u32 = 1000,
    max_depth: u32 = 10,
    /// Target mean acceptance statistic for step-size adaptation.
    delta: f64 = 0.8,
    gamma: f64 = 0.05,
    kappa: f64 = 0.75,
    t0: f64 = 10,
    init_buffer: u32 = 75,
    term_buffer: u32 = 50,
    base_window: u32 = 25,
    /// Initial values are drawn uniformly from (-r, r) on the unconstrained scale.
    init_radius: f64 = 2,
    stepsize: f64 = 1,
};

/// Per-draw sampler statistics, named as in Stan's output.
pub const Stats = struct {
    lp: f64,
    accept_stat: f64,
    stepsize: f64,
    treedepth: u32,
    n_leapfrog: u32,
    divergent: bool,
    energy: f64,
};

/// Nesterov dual averaging of log(step size) (Hoffman and Gelman 2014),
/// as in stan/mcmc/stepsize_adaptation.hpp.
pub const DualAveraging = struct {
    mu: f64,
    delta: f64,
    gamma: f64,
    kappa: f64,
    t0: f64,
    counter: f64 = 0,
    s_bar: f64 = 0,
    x_bar: f64 = 0,

    pub fn restart(self: *DualAveraging) void {
        self.counter = 0;
        self.s_bar = 0;
        self.x_bar = 0;
    }

    pub fn learn(self: *DualAveraging, epsilon: *f64, accept_stat: f64) void {
        self.counter += 1;
        const a = @min(accept_stat, 1);
        const eta = 1 / (self.counter + self.t0);
        self.s_bar = (1 - eta) * self.s_bar + eta * (self.delta - a);
        const x = self.mu - self.s_bar * @sqrt(self.counter) / self.gamma;
        const x_eta = std.math.pow(f64, self.counter, -self.kappa);
        self.x_bar = (1 - x_eta) * self.x_bar + x_eta * x;
        epsilon.* = @exp(x);
    }

    pub fn complete(self: *const DualAveraging, epsilon: *f64) void {
        if (self.counter > 0) epsilon.* = @exp(self.x_bar);
    }
};

/// Stan's warmup windows (stan/mcmc/windowed_adaptation.hpp). Counts draws
/// and says when a metric window opens and closes.
pub const Windows = struct {
    num_warmup: u32 = 0,
    init_buffer: u32 = 0,
    term_buffer: u32 = 0,
    base_window: u32 = 0,
    counter: u32 = 0,
    size: u32 = 0,
    next: u32 = std.math.maxInt(u32),

    pub fn init(num_warmup: u32, init_buffer: u32, term_buffer: u32, base_window: u32) Windows {
        var w: Windows = .{};
        if (num_warmup < 20) return w; // no metric adaptation
        if (init_buffer + base_window + term_buffer > num_warmup) {
            const nw: f64 = @floatFromInt(num_warmup);
            w.init_buffer = @intFromFloat(0.15 * nw);
            w.term_buffer = @intFromFloat(0.10 * nw);
            w.base_window = num_warmup - (w.init_buffer + w.term_buffer);
        } else {
            w.init_buffer = init_buffer;
            w.term_buffer = term_buffer;
            w.base_window = base_window;
        }
        w.num_warmup = num_warmup;
        w.restart();
        return w;
    }

    pub fn restart(w: *Windows) void {
        w.counter = 0;
        w.size = w.base_window;
        w.next = w.init_buffer + w.size -% 1;
    }

    pub fn inWindow(w: *const Windows) bool {
        return w.counter >= w.init_buffer and w.counter < w.num_warmup -% w.term_buffer and w.counter != w.num_warmup;
    }

    pub fn atWindowEnd(w: *const Windows) bool {
        return w.counter == w.next and w.counter != w.num_warmup;
    }

    pub fn computeNext(w: *Windows) void {
        const last = w.num_warmup - w.term_buffer - 1;
        if (w.next == last) return;
        w.size *= 2;
        w.next = w.counter + w.size;
        if (w.next == last) return;
        if (w.next + 2 * w.size >= w.num_warmup - w.term_buffer) w.next = last;
    }
};

/// Called after every warmup transition, for example to save warmup draws.
pub const WarmupHook = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, stats: Stats, q: []const f64) void,
};

/// Records every leapfrog state of one transition, for pictures.
pub const TracePoint = struct {
    q: [2]f64,
    p: [2]f64,
    hamiltonian: f64,
    /// Which doubling the state belongs to: 0 for the first new state.
    level: u32,
    /// +1 for forward in time, -1 for backward, 0 for the starting point.
    direction: i8,
};

pub fn Sampler(comptime M: type) type {
    const n = M.dim;
    return struct {
        const Self = @This();
        pub const V = @Vector(n, f64);

        /// A point in phase space with its cached log density and gradient.
        pub const Point = struct {
            q: V,
            p: V,
            grad: V,
            lp: f64,
        };

        model: *const M,
        prng: std.Random.DefaultPrng,
        inv_metric: V = @splat(1),
        epsilon: f64 = 1,
        max_depth: u32 = 10,
        max_delta_h: f64 = 1000,
        z: Point,

        // Statistics of the last transition.
        depth: u32 = 0,
        divergent: bool = false,
        n_leapfrog: u32 = 0,
        energy: f64 = 0,
        level: u32 = 0,

        trace: ?*std.ArrayList(TracePoint) = null,
        warmup_hook: ?WarmupHook = null,
        trace_gpa: std.mem.Allocator = undefined,
        gradient_evals: u64 = 0,

        inline fn sp(x: f64) V {
            return @splat(x);
        }

        fn uniform(self: *Self) f64 {
            return self.prng.random().float(f64);
        }

        pub fn init(model: *const M, seed: u64) Self {
            return .{
                .model = model,
                .prng = std.Random.DefaultPrng.init(seed),
                .z = .{ .q = @splat(0), .p = @splat(0), .grad = @splat(0), .lp = 0 },
            };
        }

        /// Evaluates the log density and gradient at z.q.
        pub fn evaluate(self: *Self, z: *Point) void {
            const q: [n]f64 = z.q;
            var g: [n]f64 = undefined;
            z.lp = models.logDensityGrad(M, self.model, &q, &g);
            z.grad = g;
            self.gradient_evals += 1;
        }

        /// Random initial values in (-radius, radius), retried until the log
        /// density and its gradient are finite (Stan tries 100 times).
        pub fn initialize(self: *Self, radius: f64) !void {
            for (0..100) |_| {
                var q: [n]f64 = undefined;
                for (&q) |*v| v.* = (self.uniform() * 2 - 1) * radius;
                self.z.q = q;
                self.evaluate(&self.z);
                const g: [n]f64 = self.z.grad;
                var ok = std.math.isFinite(self.z.lp);
                for (g) |v| ok = ok and std.math.isFinite(v);
                if (ok) return;
            }
            return error.InitializationFailed;
        }

        pub fn hamiltonian(self: *const Self, z: *const Point) f64 {
            return -z.lp + 0.5 * @reduce(.Add, z.p * z.p * self.inv_metric);
        }

        fn sampleMomentum(self: *Self, z: *Point) void {
            var p: [n]f64 = undefined;
            const r = self.prng.random();
            for (&p) |*v| v.* = r.floatNorm(f64);
            z.p = @as(V, p) / @sqrt(self.inv_metric);
        }

        fn leapfrog(self: *Self, z: *Point, eps: f64) void {
            z.p += sp(0.5 * eps) * z.grad;
            z.q += sp(eps) * self.inv_metric * z.p;
            self.evaluate(z);
            z.p += sp(0.5 * eps) * z.grad;
        }

        fn criterion(p_sharp_minus: V, p_sharp_plus: V, rho: V) bool {
            return @reduce(.Add, p_sharp_plus * rho) > 0 and @reduce(.Add, p_sharp_minus * rho) > 0;
        }

        fn energyAt(self: *const Self, z: *const Point) f64 {
            const h = self.hamiltonian(z);
            return if (std.math.isNan(h)) std.math.inf(f64) else h;
        }

        fn buildTree(
            self: *Self,
            depth: u32,
            z: *Point,
            z_propose: *Point,
            p_sharp_beg: *V,
            p_sharp_end: *V,
            rho: *V,
            p_beg: *V,
            p_end: *V,
            h0: f64,
            sign: f64,
            n_leapfrog: *u32,
            log_sum_weight: *f64,
            sum_metro_prob: *f64,
        ) bool {
            if (depth == 0) {
                self.leapfrog(z, sign * self.epsilon);
                n_leapfrog.* += 1;
                const h = self.energyAt(z);
                if (h - h0 > self.max_delta_h) self.divergent = true;
                log_sum_weight.* = smath.logSumExp(log_sum_weight.*, h0 - h);
                sum_metro_prob.* += if (h0 - h > 0) 1 else @exp(h0 - h);
                z_propose.* = z.*;
                p_sharp_beg.* = self.inv_metric * z.p;
                p_sharp_end.* = p_sharp_beg.*;
                rho.* += z.p;
                p_beg.* = z.p;
                p_end.* = z.p;
                if (self.trace) |t| {
                    const q: [n]f64 = z.q;
                    const p: [n]f64 = z.p;
                    t.append(self.trace_gpa, .{
                        .q = .{ q[0], if (n > 1) q[1] else 0 },
                        .p = .{ p[0], if (n > 1) p[1] else 0 },
                        .hamiltonian = h,
                        .level = self.level,
                        .direction = if (sign > 0) 1 else -1,
                    }) catch {};
                }
                return !self.divergent;
            }

            var log_sum_weight_init = -std.math.inf(f64);
            var p_init_end: V = undefined;
            var p_sharp_init_end: V = undefined;
            var rho_init: V = @splat(0);
            if (!self.buildTree(depth - 1, z, z_propose, p_sharp_beg, &p_sharp_init_end, &rho_init, p_beg, &p_init_end, h0, sign, n_leapfrog, &log_sum_weight_init, sum_metro_prob))
                return false;

            var z_propose_final = z.*;
            var log_sum_weight_final = -std.math.inf(f64);
            var p_final_beg: V = undefined;
            var p_sharp_final_beg: V = undefined;
            var rho_final: V = @splat(0);
            if (!self.buildTree(depth - 1, z, &z_propose_final, &p_sharp_final_beg, p_sharp_end, &rho_final, &p_final_beg, p_end, h0, sign, n_leapfrog, &log_sum_weight_final, sum_metro_prob))
                return false;

            // Multinomial sample from the right subtree.
            const log_sum_weight_subtree = smath.logSumExp(log_sum_weight_init, log_sum_weight_final);
            log_sum_weight.* = smath.logSumExp(log_sum_weight.*, log_sum_weight_subtree);
            if (log_sum_weight_final > log_sum_weight_subtree) {
                z_propose.* = z_propose_final;
            } else if (self.uniform() < @exp(log_sum_weight_final - log_sum_weight_subtree)) {
                z_propose.* = z_propose_final;
            }

            const rho_subtree = rho_init + rho_final;
            rho.* += rho_subtree;
            // Demand the criterion across the merged subtree and across the
            // boundary between its halves.
            var persist = criterion(p_sharp_beg.*, p_sharp_end.*, rho_subtree);
            persist = persist and criterion(p_sharp_beg.*, p_sharp_final_beg, rho_init + p_final_beg);
            persist = persist and criterion(p_sharp_init_end, p_sharp_end.*, rho_final + p_init_end);
            return persist;
        }

        /// One NUTS transition from the current state.
        pub fn transition(self: *Self) Stats {
            var z = self.z;
            self.sampleMomentum(&z);

            var z_fwd = z;
            var z_bck = z;
            var z_sample = z;
            var z_propose = z;

            var p_fwd_fwd = z.p;
            var p_sharp_fwd_fwd = self.inv_metric * z.p;
            var p_fwd_bck = z.p;
            var p_sharp_fwd_bck = p_sharp_fwd_fwd;
            var p_bck_fwd = z.p;
            var p_sharp_bck_fwd = p_sharp_fwd_fwd;
            var p_bck_bck = z.p;
            var p_sharp_bck_bck = p_sharp_fwd_fwd;

            var rho = z.p;
            var log_sum_weight: f64 = 0;
            const h0 = self.hamiltonian(&z);
            var n_leapfrog: u32 = 0;
            var sum_metro_prob: f64 = 0;
            if (self.trace) |t| {
                const q: [n]f64 = z.q;
                const p: [n]f64 = z.p;
                t.append(self.trace_gpa, .{
                    .q = .{ q[0], if (n > 1) q[1] else 0 },
                    .p = .{ p[0], if (n > 1) p[1] else 0 },
                    .hamiltonian = h0,
                    .level = 0,
                    .direction = 0,
                }) catch {};
            }

            self.depth = 0;
            self.divergent = false;

            while (self.depth < self.max_depth) {
                var rho_fwd: V = @splat(0);
                var rho_bck: V = @splat(0);
                var valid = false;
                var log_sum_weight_subtree = -std.math.inf(f64);
                self.level = self.depth;

                if (self.uniform() > 0.5) {
                    // Extend the trajectory forward.
                    z = z_fwd;
                    rho_bck = rho;
                    p_bck_fwd = p_fwd_fwd;
                    p_sharp_bck_fwd = p_sharp_fwd_fwd;
                    valid = self.buildTree(self.depth, &z, &z_propose, &p_sharp_fwd_bck, &p_sharp_fwd_fwd, &rho_fwd, &p_fwd_bck, &p_fwd_fwd, h0, 1, &n_leapfrog, &log_sum_weight_subtree, &sum_metro_prob);
                    z_fwd = z;
                } else {
                    // Extend the trajectory backward.
                    z = z_bck;
                    rho_fwd = rho;
                    p_fwd_bck = p_bck_bck;
                    p_sharp_fwd_bck = p_sharp_bck_bck;
                    valid = self.buildTree(self.depth, &z, &z_propose, &p_sharp_bck_fwd, &p_sharp_bck_bck, &rho_bck, &p_bck_fwd, &p_bck_bck, h0, -1, &n_leapfrog, &log_sum_weight_subtree, &sum_metro_prob);
                    z_bck = z;
                }

                if (!valid) break;
                self.depth += 1;

                // Biased progressive sampling between the old and new parts.
                if (log_sum_weight_subtree > log_sum_weight) {
                    z_sample = z_propose;
                } else if (self.uniform() < @exp(log_sum_weight_subtree - log_sum_weight)) {
                    z_sample = z_propose;
                }
                log_sum_weight = smath.logSumExp(log_sum_weight, log_sum_weight_subtree);

                rho = rho_bck + rho_fwd;
                var persist = criterion(p_sharp_bck_bck, p_sharp_fwd_fwd, rho);
                persist = persist and criterion(p_sharp_bck_bck, p_sharp_fwd_bck, rho_bck + p_fwd_bck);
                persist = persist and criterion(p_sharp_bck_fwd, p_sharp_fwd_fwd, rho_fwd + p_bck_fwd);
                if (!persist) break;
            }

            self.n_leapfrog = n_leapfrog;
            const accept_prob = sum_metro_prob / @as(f64, @floatFromInt(n_leapfrog));
            self.z = z_sample;
            self.energy = self.hamiltonian(&self.z);
            return .{
                .lp = self.z.lp,
                .accept_stat = accept_prob,
                .stepsize = self.epsilon,
                .treedepth = self.depth,
                .n_leapfrog = n_leapfrog,
                .divergent = self.divergent,
                .energy = self.energy,
            };
        }

        /// Stan's heuristic: double or halve the step size until the
        /// acceptance probability of one leapfrog step crosses 0.8.
        pub fn initStepsize(self: *Self) !void {
            const z_init = self.z;
            defer self.z = z_init;
            if (self.epsilon == 0 or self.epsilon > 1e7 or std.math.isNan(self.epsilon)) return;
            const log08 = @log(0.8);

            var z = z_init;
            self.sampleMomentum(&z);
            var h0 = self.hamiltonian(&z);
            self.leapfrog(&z, self.epsilon);
            const dh = h0 - self.energyAt(&z);
            const direction: i32 = if (dh > log08) 1 else -1;

            while (true) {
                z = z_init;
                self.sampleMomentum(&z);
                h0 = self.hamiltonian(&z);
                self.leapfrog(&z, self.epsilon);
                const delta_h = h0 - self.energyAt(&z);
                if (direction == 1 and !(delta_h > log08)) break;
                if (direction == -1 and !(delta_h < log08)) break;
                self.epsilon = if (direction == 1) 2 * self.epsilon else 0.5 * self.epsilon;
                if (self.epsilon > 1e7) return error.ImproperPosterior;
                if (self.epsilon == 0) return error.NoStepSize;
            }
        }

        /// Warmup with step-size and diagonal-metric adaptation.
        pub fn warmup(self: *Self, cfg: Config) !void {
            self.max_depth = cfg.max_depth;
            self.epsilon = cfg.stepsize;
            var da: DualAveraging = .{ .mu = @log(10 * cfg.stepsize), .delta = cfg.delta, .gamma = cfg.gamma, .kappa = cfg.kappa, .t0 = cfg.t0 };
            var windows = Windows.init(cfg.num_warmup, cfg.init_buffer, cfg.term_buffer, cfg.base_window);
            var count: f64 = 0;
            var mean: V = @splat(0);
            var m2: V = @splat(0);

            try self.initStepsize();
            for (0..cfg.num_warmup) |_| {
                const s = self.transition();
                if (self.warmup_hook) |h| {
                    const q: [n]f64 = self.z.q;
                    h.call(h.ctx, s, &q);
                }
                da.learn(&self.epsilon, s.accept_stat);

                // Welford variance over the draws inside the current window.
                if (windows.inWindow()) {
                    count += 1;
                    const d = self.z.q - mean;
                    mean += d / sp(count);
                    m2 += (self.z.q - mean) * d;
                }
                if (windows.atWindowEnd()) {
                    windows.computeNext();
                    const variance = m2 / sp(count - 1);
                    self.inv_metric = sp(count / (count + 5)) * variance + sp(1e-3 * 5 / (count + 5));
                    const im: [n]f64 = self.inv_metric;
                    for (im) |v| if (!std.math.isFinite(v)) return error.MetricOverflow;
                    count = 0;
                    mean = @splat(0);
                    m2 = @splat(0);
                    windows.counter += 1;
                    try self.initStepsize();
                    da.mu = @log(10 * self.epsilon);
                    da.restart();
                } else {
                    windows.counter += 1;
                }
            }
            da.complete(&self.epsilon);
        }
    };
}

// ---------------------------------------------------------------------------

const testing = std.testing;

test "warmup windows match Stan's schedule for 1000 warmup draws" {
    var w = Windows.init(1000, 75, 50, 25);
    var ends: [8]u32 = undefined;
    var k: usize = 0;
    var first_in: ?u32 = null;
    var last_in: u32 = 0;
    for (0..1000) |_| {
        if (w.inWindow()) {
            if (first_in == null) first_in = w.counter;
            last_in = w.counter;
        }
        if (w.atWindowEnd()) {
            ends[k] = w.counter;
            k += 1;
            w.computeNext();
        }
        w.counter += 1;
    }
    try testing.expectEqual(@as(?u32, 75), first_in);
    try testing.expectEqual(@as(u32, 949), last_in);
    try testing.expectEqualSlices(u32, &.{ 99, 149, 249, 449, 949 }, ends[0..k]);
}

test "dual averaging converges to a step size near the target acceptance" {
    // With accept = exp(-epsilon), the target 0.8 is epsilon = -log(0.8).
    var eps: f64 = 1;
    var da: DualAveraging = .{ .mu = @log(10.0), .delta = 0.8, .gamma = 0.05, .kappa = 0.75, .t0 = 10 };
    for (0..2000) |_| da.learn(&eps, @exp(-eps));
    da.complete(&eps);
    try testing.expectApproxEqRel(-@log(0.8), eps, 0.02);
}

test "NUTS recovers the moments of a 100-dimensional standard normal" {
    const M = models.IidNormal(100);
    const m: M = .{};
    var means: [100]f64 = @splat(0);
    var sq: [100]f64 = @splat(0);
    const chains = 4;
    const draws = 500;
    for (0..chains) |c| {
        var s = Sampler(M).init(&m, 1000 + c);
        try s.initialize(2);
        try s.warmup(.{ .num_warmup = 500 });
        for (0..draws) |_| {
            _ = s.transition();
            const q: [100]f64 = s.z.q;
            for (0..100) |i| {
                means[i] += q[i];
                sq[i] += q[i] * q[i];
            }
        }
    }
    const total: f64 = chains * draws;
    var worst_mean: f64 = 0;
    var avg_var: f64 = 0;
    for (0..100) |i| {
        const mu = means[i] / total;
        worst_mean = @max(worst_mean, @abs(mu));
        avg_var += (sq[i] / total - mu * mu) / 100;
    }
    // A mean's standard error is at most about 1/sqrt(2000) = 0.022 (NUTS
    // draws of a Gaussian mean are close to independent), so 0.1 is over four
    // standard errors for the worst of 100 coordinates. The average of the 100
    // variance estimates has a standard error near 0.005; 0.03 allows for the
    // lower efficiency of NUTS on second moments.
    try testing.expect(worst_mean < 0.1);
    try testing.expect(@abs(avg_var - 1) < 0.03);
}
