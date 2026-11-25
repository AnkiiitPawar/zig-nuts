//! The target densities: seven posteriors from posteriordb, translated by hand
//! from their Stan programs, and three synthetic targets with known answers.
//!
//! Every model works on the unconstrained scale, as Stan does: a parameter
//! with bounds is a transform of a free real number, and the log density
//! includes the log Jacobian of that transform. Terms that do not depend on the
//! parameters are dropped, as Stan's `~` statements drop them, so `lp__`
//! differs from Stan's by a constant.
//!
//! A model is a struct with
//!   name, data_file   identity and the posteriordb data file (or null)
//!   dim               the number of unconstrained parameters (comptime)
//!   out_names         names of the constrained outputs, in posteriordb order
//!   Data, init        the JSON data layout and a constructor
//!   logDensity(S, x)  the log density, generic over the scalar type S, which
//!                     is `Dual(dim)` for value and gradient or `Real` for value
//!   write(u, out)     the constrained outputs for one unconstrained point
//!
//! The parameter dimension is a compile-time constant because the gradient is
//! a `@Vector(dim, f64)`. Data sizes are read at run time.

const std = @import("std");
const dual = @import("dual.zig");
const Real = dual.Real;
const Allocator = std.mem.Allocator;

pub const DataError = error{DataShape};

fn indexed(comptime base: []const u8, comptime count: usize) [count][]const u8 {
    @setEvalBranchQuota(200_000);
    var out: [count][]const u8 = undefined;
    for (0..count) |i| out[i] = std.fmt.comptimePrint("{s}[{d}]", .{ base, i + 1 });
    return out;
}

const NoData = struct {};

// ---------------------------------------------------------------------------
// posteriordb: eight_schools-eight_schools_noncentered

pub const EightSchools = struct {
    pub const name = "eight_schools_noncentered";
    pub const data_file: ?[]const u8 = "eight_schools.json";
    const J = 8;
    pub const dim = J + 2; // theta_trans[1..8], mu, log(tau)
    pub const out_names = indexed("theta", J) ++ [_][]const u8{ "mu", "tau" };
    pub const Data = struct { J: usize, y: []const f64, sigma: []const f64 };

    y: [J]f64,
    sigma: [J]f64,

    pub fn init(_: Allocator, d: Data) DataError!EightSchools {
        if (d.J != J or d.y.len != J or d.sigma.len != J) return error.DataShape;
        var m: EightSchools = undefined;
        @memcpy(&m.y, d.y);
        @memcpy(&m.sigma, d.sigma);
        return m;
    }

    pub fn logDensity(m: *const EightSchools, comptime S: type, x: *const [dim]S) S {
        const mu = x[J];
        const log_tau = x[J + 1];
        const tau = log_tau.exp();
        var lp = log_tau; // log Jacobian of tau = exp(u)
        for (0..J) |j| {
            const t = x[j];
            lp = lp.sub(t.square().scale(0.5)); // theta_trans ~ normal(0, 1)
            const theta = t.mul(tau).add(mu);
            const z = theta.addC(-m.y[j]).scale(1 / m.sigma[j]);
            lp = lp.sub(z.square().scale(0.5)); // y ~ normal(theta, sigma)
        }
        lp = lp.sub(mu.square().scale(0.5 / 25.0)); // mu ~ normal(0, 5)
        lp = lp.sub(tau.scale(0.2).square().log1p()); // tau ~ cauchy(0, 5)
        return lp;
    }

    pub fn write(_: *const EightSchools, u: *const [dim]f64, out: []f64) void {
        const tau = @exp(u[J + 1]);
        for (0..J) |j| out[j] = u[j] * tau + u[J];
        out[J] = u[J];
        out[J + 1] = tau;
    }
};

// ---------------------------------------------------------------------------
// posteriordb: arK-arK, an autoregressive model of order K = 5

pub const ArK = struct {
    pub const name = "arK";
    pub const data_file: ?[]const u8 = "arK.json";
    const K = 5;
    pub const dim = K + 2; // alpha, beta[1..K], log(sigma)
    pub const out_names = [_][]const u8{"alpha"} ++ indexed("beta", K) ++ [_][]const u8{"sigma"};
    pub const Data = struct { K: usize, T: usize, y: []const f64 };

    y: []const f64,

    pub fn init(_: Allocator, d: Data) DataError!ArK {
        if (d.K != K or d.y.len != d.T or d.T <= K) return error.DataShape;
        return .{ .y = d.y };
    }

    pub fn logDensity(m: *const ArK, comptime S: type, x: *const [dim]S) S {
        const alpha = x[0];
        const log_sigma = x[K + 1];
        const sigma = log_sigma.exp();
        var lp = log_sigma; // log Jacobian
        lp = lp.sub(alpha.square().scale(0.5 / 100.0)); // alpha ~ normal(0, 10)
        for (1..K + 1) |k| lp = lp.sub(x[k].square().scale(0.5 / 100.0)); // beta ~ normal(0, 10)
        lp = lp.sub(sigma.scale(0.4).square().log1p()); // sigma ~ cauchy(0, 2.5)
        // y[t] ~ normal(alpha + sum_k beta[k] y[t-k], sigma) for t > K
        var ss = S.constant(0);
        for (K..m.y.len) |t| {
            var mu = alpha;
            for (1..K + 1) |k| mu = x[k].mulAdd(m.y[t - k], mu);
            ss = ss.add(mu.addC(-m.y[t]).square());
        }
        const count: f64 = @floatFromInt(m.y.len - K);
        lp = lp.sub(log_sigma.scale(count));
        lp = lp.sub(ss.mul(sigma.square().recip()).scale(0.5));
        return lp;
    }

    pub fn write(_: *const ArK, u: *const [dim]f64, out: []f64) void {
        for (0..K + 1) |i| out[i] = u[i];
        out[K + 1] = @exp(u[K + 1]);
    }
};

// ---------------------------------------------------------------------------
// posteriordb: gp_pois_regr-gp_pois_regr, a latent Gaussian process with a
// Poisson likelihood and a Cholesky factor computed inside the density

pub const GpPoisRegr = struct {
    pub const name = "gp_pois_regr";
    pub const data_file: ?[]const u8 = "gp_pois_regr.json";
    const N = 11;
    pub const dim = N + 2; // log(rho), log(alpha), f_tilde[1..N]
    pub const out_names = [_][]const u8{ "rho", "alpha" } ++ indexed("f", N);
    pub const Data = struct { N: usize, x: []const f64, k: []const i64 };

    x: [N]f64,
    k: [N]f64,

    pub fn init(_: Allocator, d: Data) DataError!GpPoisRegr {
        if (d.N != N or d.x.len != N or d.k.len != N) return error.DataShape;
        var m: GpPoisRegr = undefined;
        @memcpy(&m.x, d.x);
        for (&m.k, d.k) |*o, v| o.* = @floatFromInt(v);
        return m;
    }

    /// f = L f_tilde, where L is the Cholesky factor of the covariance
    /// alpha^2 exp(-(x_i - x_j)^2 / (2 rho^2)) + 1e-10 I.
    fn latent(m: *const GpPoisRegr, comptime S: type, rho: S, alpha: S, f_tilde: *const [N]S, f: *[N]S) void {
        const alpha2 = alpha.square();
        const c = rho.square().recip().scale(-0.5);
        var L: [N][N]S = undefined;
        for (0..N) |i| {
            for (0..i + 1) |j| {
                const d = m.x[i] - m.x[j];
                var s = if (i == j) alpha2.addC(1e-10) else alpha2.mul(c.scale(d * d).exp());
                for (0..j) |r| s = s.sub(L[i][r].mul(L[j][r]));
                L[i][j] = if (i == j) s.sqrt() else s.div(L[j][j]);
            }
        }
        for (0..N) |i| {
            var acc = S.constant(0);
            for (0..i + 1) |j| acc = acc.add(L[i][j].mul(f_tilde[j]));
            f[i] = acc;
        }
    }

    pub fn logDensity(m: *const GpPoisRegr, comptime S: type, x: *const [dim]S) S {
        const log_rho = x[0];
        const log_alpha = x[1];
        const rho = log_rho.exp();
        const alpha = log_alpha.exp();
        var lp = log_rho.add(log_alpha); // log Jacobians
        lp = lp.add(log_rho.scale(24)).sub(rho.scale(4)); // rho ~ gamma(25, 4)
        lp = lp.sub(alpha.square().scale(0.125)); // alpha ~ normal(0, 2)
        const f_tilde = x[2..][0..N];
        for (f_tilde) |ft| lp = lp.sub(ft.square().scale(0.5)); // f_tilde ~ normal(0, 1)
        var f: [N]S = undefined;
        m.latent(S, rho, alpha, f_tilde, &f);
        for (f, m.k) |fi, ki| lp = lp.add(fi.scale(ki).sub(fi.exp())); // k ~ poisson_log(f)
        return lp;
    }

    pub fn write(m: *const GpPoisRegr, u: *const [dim]f64, out: []f64) void {
        const rho = Real{ .val = @exp(u[0]) };
        const alpha = Real{ .val = @exp(u[1]) };
        var ft: [N]Real = undefined;
        for (&ft, 0..) |*o, i| o.* = .{ .val = u[2 + i] };
        var f: [N]Real = undefined;
        m.latent(Real, rho, alpha, &ft, &f);
        out[0] = rho.val;
        out[1] = alpha.val;
        for (0..N) |i| out[2 + i] = f[i].val;
    }
};

// ---------------------------------------------------------------------------
// posteriordb: low_dim_gauss_mix-low_dim_gauss_mix, a two-component mixture

pub const LowDimGaussMix = struct {
    pub const name = "low_dim_gauss_mix";
    pub const data_file: ?[]const u8 = "low_dim_gauss_mix.json";
    pub const dim = 5; // ordered mu (2), log(sigma) (2), logit(theta)
    pub const out_names = indexed("mu", 2) ++ indexed("sigma", 2) ++ [_][]const u8{"theta"};
    pub const Data = struct { N: usize, y: []const f64 };

    y: []const f64,

    pub fn init(_: Allocator, d: Data) DataError!LowDimGaussMix {
        if (d.y.len != d.N) return error.DataShape;
        return .{ .y = d.y };
    }

    pub fn logDensity(m: *const LowDimGaussMix, comptime S: type, x: *const [dim]S) S {
        const mu1 = x[0];
        const mu2 = x[0].add(x[1].exp()); // ordered: mu2 = mu1 + exp(u)
        const ls1 = x[2];
        const ls2 = x[3];
        const s1 = ls1.exp();
        const s2 = ls2.exp();
        const log_theta = x[4].logInvLogit();
        const log1m_theta = x[4].log1mInvLogit();
        var lp = x[1].add(ls1).add(ls2).add(log_theta).add(log1m_theta); // log Jacobians
        lp = lp.sub(s1.square().add(s2.square()).scale(0.125)); // sigma ~ normal(0, 2)
        lp = lp.sub(mu1.square().add(mu2.square()).scale(0.125)); // mu ~ normal(0, 2)
        lp = lp.add(log_theta.add(log1m_theta).scale(4)); // theta ~ beta(5, 5)
        // target += log_mix(theta, normal_lpdf(y | mu1, sigma1), normal_lpdf(y | mu2, sigma2))
        const c = -0.5 * @log(2 * std.math.pi);
        const inv1 = s1.recip();
        const inv2 = s2.recip();
        const a0 = log_theta.sub(ls1).addC(c);
        const b0 = log1m_theta.sub(ls2).addC(c);
        var ll = S.constant(0);
        for (m.y) |yn| {
            const z1 = mu1.rsubC(yn).mul(inv1);
            const z2 = mu2.rsubC(yn).mul(inv2);
            const a = a0.sub(z1.square().scale(0.5));
            const b = b0.sub(z2.square().scale(0.5));
            ll = ll.add(a.logSumExp(b));
        }
        return lp.add(ll);
    }

    pub fn write(_: *const LowDimGaussMix, u: *const [dim]f64, out: []f64) void {
        out[0] = u[0];
        out[1] = u[0] + @exp(u[1]);
        out[2] = @exp(u[2]);
        out[3] = @exp(u[3]);
        out[4] = Real.invLogit(.{ .val = u[4] }).val;
    }
};

// ---------------------------------------------------------------------------
// posteriordb: wells_data-wells_dist, logistic regression on 3020 households

pub const WellsDist = struct {
    pub const name = "wells_dist";
    pub const data_file: ?[]const u8 = "wells_data.json";
    pub const dim = 2;
    pub const out_names = indexed("beta", 2);
    pub const Data = struct { N: usize, switched: []const i64, dist: []const f64 };

    switched: []const bool,
    dist: []const f64,

    pub fn init(arena: Allocator, d: Data) (DataError || Allocator.Error)!WellsDist {
        if (d.switched.len != d.N or d.dist.len != d.N) return error.DataShape;
        const s = try arena.alloc(bool, d.N);
        for (s, d.switched) |*o, v| o.* = v == 1;
        return .{ .switched = s, .dist = d.dist };
    }

    pub fn logDensity(m: *const WellsDist, comptime S: type, x: *const [dim]S) S {
        // switched ~ bernoulli_logit(beta[1] + beta[2] * dist); flat prior
        var lp = S.constant(0);
        for (m.switched, m.dist) |y, d| {
            const eta = x[1].mulAdd(d, x[0]);
            lp = lp.add(if (y) eta.logInvLogit() else eta.log1mInvLogit());
        }
        return lp;
    }

    pub fn write(_: *const WellsDist, u: *const [dim]f64, out: []f64) void {
        out[0] = u[0];
        out[1] = u[1];
    }
};

// ---------------------------------------------------------------------------
// posteriordb: garch-garch11, a GARCH(1,1) volatility model

pub const Garch11 = struct {
    pub const name = "garch11";
    pub const data_file: ?[]const u8 = "garch.json";
    pub const dim = 4; // mu, log(alpha0), logit(alpha1), logit of beta1 / (1 - alpha1)
    pub const out_names = [_][]const u8{ "mu", "alpha0", "alpha1", "beta1" };
    pub const Data = struct { T: usize, y: []const f64, sigma1: f64 };

    y: []const f64,
    sigma1: f64,

    pub fn init(_: Allocator, d: Data) DataError!Garch11 {
        if (d.y.len != d.T or d.T < 2) return error.DataShape;
        return .{ .y = d.y, .sigma1 = d.sigma1 };
    }

    pub fn logDensity(m: *const Garch11, comptime S: type, x: *const [dim]S) S {
        const mu = x[0];
        const alpha0 = x[1].exp();
        const alpha1 = x[2].invLogit();
        const one_minus_alpha1 = x[2].neg().invLogit();
        const beta1 = one_minus_alpha1.mul(x[3].invLogit()); // beta1 in (0, 1 - alpha1)
        // log Jacobians: alpha0 = exp(u); alpha1 = inv_logit(u);
        // beta1 = (1 - alpha1) inv_logit(u) contributes log(1 - alpha1) + log s + log(1 - s)
        var lp = x[1].add(x[2].logInvLogit()).add(x[2].log1mInvLogit());
        lp = lp.add(x[2].log1mInvLogit()).add(x[3].logInvLogit()).add(x[3].log1mInvLogit());
        // y ~ normal(mu, sigma), sigma[t]^2 = alpha0 + alpha1 (y[t-1] - mu)^2 + beta1 sigma[t-1]^2
        var v = S.constant(m.sigma1 * m.sigma1);
        lp = lp.sub(mu.rsubC(m.y[0]).square().scale(0.5 / (m.sigma1 * m.sigma1)));
        for (1..m.y.len) |t| {
            v = alpha0.add(alpha1.mul(mu.rsubC(m.y[t - 1]).square())).add(beta1.mul(v));
            const r = mu.rsubC(m.y[t]);
            lp = lp.sub(v.log().add(r.square().div(v)).scale(0.5));
        }
        return lp;
    }

    pub fn write(_: *const Garch11, u: *const [dim]f64, out: []f64) void {
        const a1 = Real.invLogit(.{ .val = u[2] }).val;
        out[0] = u[0];
        out[1] = @exp(u[1]);
        out[2] = a1;
        out[3] = Real.invLogit(.{ .val = -u[2] }).val * Real.invLogit(.{ .val = u[3] }).val;
    }
};

// ---------------------------------------------------------------------------
// posteriordb: dogs-dogs, a learning model for 30 dogs over 25 trials

pub const Dogs = struct {
    pub const name = "dogs";
    pub const data_file: ?[]const u8 = "dogs.json";
    pub const dim = 3;
    pub const out_names = indexed("beta", 3);
    pub const Data = struct { n_dogs: usize, n_trials: usize, y: []const []const i64 };

    n_dogs: usize,
    n_trials: usize,
    y: []const bool, // row-major n_dogs x n_trials

    pub fn init(arena: Allocator, d: Data) (DataError || Allocator.Error)!Dogs {
        if (d.y.len != d.n_dogs or d.n_trials < 1) return error.DataShape;
        const y = try arena.alloc(bool, d.n_dogs * d.n_trials);
        for (d.y, 0..) |row, j| {
            if (row.len != d.n_trials) return error.DataShape;
            for (row, 0..) |v, t| y[j * d.n_trials + t] = v == 1;
        }
        return .{ .n_dogs = d.n_dogs, .n_trials = d.n_trials, .y = y };
    }

    pub fn logDensity(m: *const Dogs, comptime S: type, x: *const [dim]S) S {
        var lp = x[0].square().add(x[1].square()).add(x[2].square()).scale(-0.5 / 10000.0); // beta ~ normal(0, 100)
        for (0..m.n_dogs) |j| {
            const row = m.y[j * m.n_trials ..][0..m.n_trials];
            // n_avoid and n_shock depend on data only, so they stay plain numbers
            var n_avoid: f64 = 0;
            var n_shock: f64 = 0;
            for (row, 0..) |y, t| {
                if (t > 0) {
                    if (row[t - 1]) n_shock += 1 else n_avoid += 1;
                }
                const p = x[1].mulAdd(n_avoid, x[2].mulAdd(n_shock, x[0]));
                lp = lp.add(if (y) p.logInvLogit() else p.log1mInvLogit());
            }
        }
        return lp;
    }

    pub fn write(_: *const Dogs, u: *const [dim]f64, out: []f64) void {
        for (0..dim) |i| out[i] = u[i];
    }
};

// ---------------------------------------------------------------------------
// Synthetic targets with exact answers

/// A standard normal in `n` dimensions.
pub fn IidNormal(comptime n: usize) type {
    return struct {
        const Self = @This();
        pub const name = std.fmt.comptimePrint("iid_normal_{d}", .{n});
        pub const data_file: ?[]const u8 = null;
        pub const dim = n;
        pub const out_names = indexed("x", n);
        pub const Data = NoData;

        pub fn init(_: Allocator, _: Data) DataError!Self {
            return .{};
        }

        pub fn logDensity(_: *const Self, comptime S: type, x: *const [dim]S) S {
            var lp = S.constant(0);
            for (x) |xi| lp = lp.sub(xi.square().scale(0.5));
            return lp;
        }

        pub fn write(_: *const Self, u: *const [dim]f64, out: []f64) void {
            @memcpy(out[0..dim], u);
        }
    };
}

/// Neal's funnel: v ~ normal(0, 3), x[i] ~ normal(0, exp(v / 2)), in the
/// centred form that defeats a sampler with a single step size.
pub fn Funnel(comptime n: usize) type {
    return struct {
        const Self = @This();
        pub const name = std.fmt.comptimePrint("funnel_{d}", .{n});
        pub const data_file: ?[]const u8 = null;
        pub const dim = n;
        pub const out_names = [_][]const u8{"v"} ++ indexed("x", n - 1);
        pub const Data = NoData;

        pub fn init(_: Allocator, _: Data) DataError!Self {
            return .{};
        }

        pub fn logDensity(_: *const Self, comptime S: type, x: *const [dim]S) S {
            const v = x[0];
            var ss = S.constant(0);
            for (x[1..]) |xi| ss = ss.add(xi.square());
            var lp = v.square().scale(-0.5 / 9.0);
            lp = lp.sub(v.scale(0.5 * @as(f64, @floatFromInt(n - 1))));
            lp = lp.sub(ss.mul(v.neg().exp()).scale(0.5));
            return lp;
        }

        pub fn write(_: *const Self, u: *const [dim]f64, out: []f64) void {
            @memcpy(out[0..dim], u);
        }
    };
}

/// A bivariate normal with unit variances and correlation 0.8. It is used to
/// draw a single trajectory, where the U-turn is easy to see.
pub const CorrNormal = struct {
    pub const name = "corr_normal_2";
    pub const data_file: ?[]const u8 = null;
    pub const dim = 2;
    pub const out_names = indexed("x", 2);
    pub const Data = NoData;
    pub const rho = 0.8;

    pub fn init(_: Allocator, _: Data) DataError!CorrNormal {
        return .{};
    }

    pub fn logDensity(_: *const CorrNormal, comptime S: type, x: *const [dim]S) S {
        const q = x[0].square().sub(x[0].mul(x[1]).scale(2 * rho)).add(x[1].square());
        return q.scale(-0.5 / (1 - rho * rho));
    }

    pub fn write(_: *const CorrNormal, u: *const [dim]f64, out: []f64) void {
        out[0] = u[0];
        out[1] = u[1];
    }
};

/// Every model, in the order they are listed by the command-line tool.
pub const all = .{ EightSchools, ArK, GpPoisRegr, LowDimGaussMix, WellsDist, Garch11, Dogs, IidNormal(100), Funnel(10), CorrNormal };

/// Parses a model's data from JSON. Memory comes from `arena`.
pub fn load(comptime M: type, arena: Allocator, json: []const u8) !M {
    const d = try std.json.parseFromSliceLeaky(M.Data, arena, json, .{ .ignore_unknown_fields = true });
    return M.init(arena, d);
}

/// The log density and its gradient at the unconstrained point `q`.
pub fn logDensityGrad(comptime M: type, m: *const M, q: *const [M.dim]f64, grad: *[M.dim]f64) f64 {
    const D = dual.Dual(M.dim);
    const x = D.variables(q);
    const lp = m.logDensity(D, &x);
    grad.* = lp.grad;
    return lp.val;
}

/// The log density alone.
pub fn logDensityValue(comptime M: type, m: *const M, q: *const [M.dim]f64) f64 {
    var x: [M.dim]Real = undefined;
    for (&x, q) |*o, v| o.* = .{ .val = v };
    return m.logDensity(Real, &x).val;
}

// ---------------------------------------------------------------------------

const testing = std.testing;
const pdb = @import("pdb_data");

test "every gradient matches central finite differences" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    inline for (all) |M| {
        const json = if (M.data_file) |f| pdb.get(f) orelse return error.MissingData else "{}";
        const m = try load(M, arena, json);
        for (0..5) |_| {
            var q: [M.dim]f64 = undefined;
            for (&q) |*v| v.* = rand.float(f64) * 2 - 1;
            var g: [M.dim]f64 = undefined;
            const lp = logDensityGrad(M, &m, &q, &g);
            try testing.expect(std.math.isFinite(lp));
            try testing.expectApproxEqRel(logDensityValue(M, &m, &q), lp, 1e-12);
            for (0..M.dim) |i| {
                const h = 1e-5 * @max(1, @abs(q[i]));
                var qp = q;
                var qm = q;
                qp[i] += h;
                qm[i] -= h;
                const fd = (logDensityValue(M, &m, &qp) - logDensityValue(M, &m, &qm)) / (2 * h);
                const tol = 1e-6 * (1 + @abs(g[i]) + @abs(lp) * 1e-3);
                if (!(@abs(fd - g[i]) <= tol)) {
                    std.debug.print("{s}: d/dq[{d}] = {d}, finite difference {d}\n", .{ M.name, i, g[i], fd });
                    return error.GradientMismatch;
                }
            }
        }
    }
}

test "output names match the output count" {
    inline for (all) |M| {
        var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena_state.deinit();
        const json = if (M.data_file) |f| pdb.get(f) orelse return error.MissingData else "{}";
        const m = try load(M, arena_state.allocator(), json);
        var u: [M.dim]f64 = @splat(0.1);
        var out: [M.out_names.len]f64 = undefined;
        m.write(&u, &out);
        for (out) |v| try testing.expect(std.math.isFinite(v));
    }
}

test "eight schools: constrained outputs follow the transform" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const m = try load(EightSchools, arena_state.allocator(), pdb.get("eight_schools.json").?);
    var u: [EightSchools.dim]f64 = @splat(0);
    u[0] = 1.5;
    u[8] = 2; // mu
    u[9] = @log(3.0); // tau
    var out: [10]f64 = undefined;
    m.write(&u, &out);
    try testing.expectApproxEqAbs(@as(f64, 1.5 * 3 + 2), out[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2), out[1], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3), out[9], 1e-12);
}
