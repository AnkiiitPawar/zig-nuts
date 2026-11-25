//! Convergence diagnostics from Vehtari, Gelman, Simpson, Carpenter and
//! Bürkner (2021), "Rank-normalization, folding, and localization: an improved
//! R-hat for assessing convergence of MCMC", following the definitions used by
//! ArviZ and the posterior R package:
//!
//!   rhatRank  max of the split R-hat of rank-normalised draws and of
//!             rank-normalised folded draws |x - median|
//!   essBulk   effective sample size of rank-normalised split chains
//!   essTail   the smaller of the ESS of the indicators x <= q05 and x <= q95
//!   essMean   effective sample size of the raw split chains (for MCSE)
//!
//! Draws are laid out chain by chain: x[c * n_draw + i].

const std = @import("std");
const smath = @import("math.zig");
const Allocator = std.mem.Allocator;

/// A set of chains of equal length, stored contiguously.
const Chains = struct {
    data: []f64,
    m: usize,
    n: usize,

    fn chain(c: Chains, k: usize) []f64 {
        return c.data[k * c.n ..][0..c.n];
    }
};

/// Splits each chain into its first and last halves (dropping the middle
/// draw when the length is odd), giving twice as many chains.
fn split(gpa: Allocator, x: []const f64, m: usize) !Chains {
    const n = x.len / m;
    const half = n / 2;
    const out = try gpa.alloc(f64, 2 * m * half);
    for (0..m) |c| {
        const src = x[c * n ..][0..n];
        @memcpy(out[(2 * c) * half ..][0..half], src[0..half]);
        @memcpy(out[(2 * c + 1) * half ..][0..half], src[n - half ..]);
    }
    return .{ .data = out, .m = 2 * m, .n = half };
}

/// Replaces values by normal scores of their ranks: average ranks for ties,
/// then the quantile of (rank - 3/8) / (S + 1/4).
fn zScale(gpa: Allocator, values: []f64) !void {
    const s = values.len;
    const idx = try gpa.alloc(usize, s);
    defer gpa.free(idx);
    for (idx, 0..) |*v, i| v.* = i;
    std.sort.pdq(usize, idx, values, struct {
        fn lt(v: []f64, a: usize, b: usize) bool {
            return v[a] < v[b];
        }
    }.lt);
    const ranks = try gpa.alloc(f64, s);
    defer gpa.free(ranks);
    var i: usize = 0;
    while (i < s) {
        var j = i + 1;
        while (j < s and values[idx[j]] == values[idx[i]]) j += 1;
        // Positions i..j-1 hold equal values; their ranks are i+1..j.
        const avg = @as(f64, @floatFromInt(i + 1 + j)) / 2;
        for (idx[i..j]) |k| ranks[k] = avg;
        i = j;
    }
    const sf: f64 = @floatFromInt(s);
    for (values, ranks) |*v, r| v.* = smath.normalQuantile((r - 0.375) / (sf + 0.25));
}

fn mean(x: []const f64) f64 {
    var acc: f64 = 0;
    for (x) |v| acc += v;
    return acc / @as(f64, @floatFromInt(x.len));
}

fn variance(x: []const f64) f64 {
    const mu = mean(x);
    var acc: f64 = 0;
    for (x) |v| acc += (v - mu) * (v - mu);
    return acc / @as(f64, @floatFromInt(x.len - 1));
}

/// Split R-hat of chains that are already split.
fn rhatOf(c: Chains) f64 {
    var means_buf: [512]f64 = undefined;
    const means = means_buf[0..c.m];
    var w: f64 = 0;
    for (0..c.m) |k| {
        means[k] = mean(c.chain(k));
        w += variance(c.chain(k));
    }
    w /= @floatFromInt(c.m);
    const nf: f64 = @floatFromInt(c.n);
    const b = nf * variance(means);
    return @sqrt((b / w + nf - 1) / nf);
}

/// Mean over chains of the autocovariance at `lag` (divided by n, as the FFT
/// estimator does).
fn meanAutocov(c: Chains, means: []const f64, lag: usize) f64 {
    var total: f64 = 0;
    for (0..c.m) |k| {
        const x = c.chain(k);
        const mu = means[k];
        var acc: f64 = 0;
        for (0..c.n - lag) |i| acc += (x[i] - mu) * (x[i + lag] - mu);
        total += acc / @as(f64, @floatFromInt(c.n));
    }
    return total / @as(f64, @floatFromInt(c.m));
}

/// Effective sample size of chains that are already split, with Geyer's
/// initial monotone sequence.
fn essOf(gpa: Allocator, c: Chains) !f64 {
    const s: f64 = @floatFromInt(c.m * c.n);
    var lo = c.data[0];
    var hi = c.data[0];
    for (c.data) |v| {
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    if (hi - lo < 1e-15) return s;

    const means = try gpa.alloc(f64, c.m);
    defer gpa.free(means);
    for (means, 0..) |*v, k| v.* = mean(c.chain(k));

    const nf: f64 = @floatFromInt(c.n);
    const mean_var = meanAutocov(c, means, 0) * nf / (nf - 1);
    var var_plus = mean_var * (nf - 1) / nf;
    if (c.m > 1) var_plus += variance(means);

    const rho = try gpa.alloc(f64, c.n);
    defer gpa.free(rho);
    @memset(rho, 0);
    var rho_even: f64 = 1;
    rho[0] = rho_even;
    var rho_odd = 1 - (mean_var - meanAutocov(c, means, 1)) / var_plus;
    rho[1] = rho_odd;

    var t: usize = 1;
    while (t + 3 < c.n and rho_even + rho_odd > 0) {
        rho_even = 1 - (mean_var - meanAutocov(c, means, t + 1)) / var_plus;
        rho_odd = 1 - (mean_var - meanAutocov(c, means, t + 2)) / var_plus;
        if (rho_even + rho_odd >= 0) {
            rho[t + 1] = rho_even;
            rho[t + 2] = rho_odd;
        }
        t += 2;
    }
    const max_t = t - 2;
    if (rho_even > 0) rho[max_t + 1] = rho_even;

    t = 1;
    while (t + 2 <= max_t) {
        if (rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]) {
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2;
            rho[t + 2] = rho[t + 1];
        }
        t += 2;
    }

    var sum: f64 = 0;
    for (rho[0 .. max_t + 1]) |v| sum += v;
    var tau = -1 + 2 * sum + rho[max_t + 1];
    tau = @max(tau, 1 / std.math.log10(s));
    for (rho) |v| if (std.math.isNan(v)) return std.math.nan(f64);
    return s / tau;
}

/// The type 7 (linear interpolation) quantile of all draws.
pub fn quantile(gpa: Allocator, x: []const f64, p: f64) !f64 {
    const sorted = try gpa.dupe(f64, x);
    defer gpa.free(sorted);
    std.sort.pdq(f64, sorted, {}, std.sort.asc(f64));
    return quantileSorted(sorted, p);
}

fn quantileSorted(sorted: []const f64, p: f64) f64 {
    const h = @as(f64, @floatFromInt(sorted.len - 1)) * p;
    const lo: usize = @intFromFloat(@floor(h));
    const hi = @min(lo + 1, sorted.len - 1);
    return sorted[lo] + (h - @floor(h)) * (sorted[hi] - sorted[lo]);
}

pub fn rhatRank(gpa: Allocator, x: []const f64, m: usize) !f64 {
    const sc = try split(gpa, x, m);
    defer gpa.free(sc.data);
    const folded = try gpa.dupe(f64, sc.data);
    defer gpa.free(folded);

    const med = try quantile(gpa, sc.data, 0.5);
    for (folded) |*v| v.* = @abs(v.* - med);

    try zScale(gpa, sc.data);
    const bulk = rhatOf(sc);
    try zScale(gpa, folded);
    const tail = rhatOf(.{ .data = folded, .m = sc.m, .n = sc.n });
    return @max(bulk, tail);
}

pub fn essBulk(gpa: Allocator, x: []const f64, m: usize) !f64 {
    const sc = try split(gpa, x, m);
    defer gpa.free(sc.data);
    try zScale(gpa, sc.data);
    return essOf(gpa, sc);
}

pub fn essMean(gpa: Allocator, x: []const f64, m: usize) !f64 {
    const sc = try split(gpa, x, m);
    defer gpa.free(sc.data);
    return essOf(gpa, sc);
}

fn essQuantile(gpa: Allocator, x: []const f64, m: usize, p: f64) !f64 {
    const q = try quantile(gpa, x, p);
    const ind = try gpa.alloc(f64, x.len);
    defer gpa.free(ind);
    for (ind, x) |*o, v| o.* = if (v <= q) 1 else 0;
    const sc = try split(gpa, ind, m);
    defer gpa.free(sc.data);
    return essOf(gpa, sc);
}

pub fn essTail(gpa: Allocator, x: []const f64, m: usize) !f64 {
    return @min(try essQuantile(gpa, x, m, 0.05), try essQuantile(gpa, x, m, 0.95));
}

/// Summary of one quantity over all chains.
pub const Summary = struct {
    mean: f64,
    sd: f64,
    q5: f64,
    q50: f64,
    q95: f64,
    ess_bulk: f64,
    ess_tail: f64,
    rhat: f64,
};

pub fn summarize(gpa: Allocator, x: []const f64, m: usize) !Summary {
    const sorted = try gpa.dupe(f64, x);
    defer gpa.free(sorted);
    std.sort.pdq(f64, sorted, {}, std.sort.asc(f64));
    return .{
        .mean = mean(x),
        .sd = @sqrt(variance(x)),
        .q5 = quantileSorted(sorted, 0.05),
        .q50 = quantileSorted(sorted, 0.5),
        .q95 = quantileSorted(sorted, 0.95),
        .ess_bulk = try essBulk(gpa, x, m),
        .ess_tail = try essTail(gpa, x, m),
        .rhat = try rhatRank(gpa, x, m),
    };
}

// ---------------------------------------------------------------------------

const testing = std.testing;

/// A deterministic test sequence: 4 chains of 100 draws of an AR(1)-like
/// recurrence driven by a fixed linear congruential generator.
fn fixture(buf: *[400]f64) void {
    var state: u64 = 12345;
    for (0..4) |c| {
        var prev: f64 = @as(f64, @floatFromInt(c)) * 0.1;
        for (0..100) |i| {
            state = state *% 6364136223846793005 +% 1442695040888963407;
            const u = @as(f64, @floatFromInt(state >> 11)) / 9007199254740992.0;
            prev = 0.6 * prev + (u - 0.5);
            buf[c * 100 + i] = prev;
        }
    }
}

test "diagnostics agree with ArviZ on a fixed sequence" {
    var buf: [400]f64 = undefined;
    fixture(&buf);
    // Expected values computed by arviz-stats 1.3.3 (array_stats._ess_bulk,
    // _ess_tail, _ess_mean and _rhat_rank) on the same sequence; see
    // scripts/check_diagnostics.py.
    const gpa = testing.allocator;
    try testing.expectApproxEqRel(@as(f64, 148.2803365917968), try essBulk(gpa, &buf, 4), 1e-10);
    try testing.expectApproxEqRel(@as(f64, 186.15103495601377), try essTail(gpa, &buf, 4), 1e-10);
    try testing.expectApproxEqRel(@as(f64, 148.58258478165308), try essMean(gpa, &buf, 4), 1e-10);
    try testing.expectApproxEqRel(@as(f64, 1.0108962737888763), try rhatRank(gpa, &buf, 4), 1e-12);
}

test "independent draws have ESS near the number of draws" {
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    var x: [4000]f64 = undefined;
    for (&x) |*v| v.* = r.floatNorm(f64);
    const e = try essBulk(testing.allocator, &x, 4);
    try testing.expect(e > 3400 and e < 4600);
    const rh = try rhatRank(testing.allocator, &x, 4);
    try testing.expect(rh < 1.01);
}

test "constant draws have ESS equal to the number of draws" {
    const x: [40]f64 = @splat(3.0);
    try testing.expectEqual(@as(f64, 40), try essMean(testing.allocator, &x, 4));
}
