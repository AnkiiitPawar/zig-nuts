//! Scalar helpers shared by the models, the sampler and the diagnostics.

const std = @import("std");

/// log(1 + exp(x)) without overflow or loss of precision.
pub fn log1pExp(x: f64) f64 {
    if (x > 0) return x + std.math.log1p(@exp(-x));
    return std.math.log1p(@exp(x));
}

/// 1 / (1 + exp(-x)) without overflow.
pub fn invLogit(x: f64) f64 {
    if (x >= 0) return 1 / (1 + @exp(-x));
    const e = @exp(x);
    return e / (1 + e);
}

/// log(exp(a) + exp(b)) without overflow.
pub fn logSumExp(a: f64, b: f64) f64 {
    const m = @max(a, b);
    if (m == -std.math.inf(f64)) return m;
    if (m == std.math.inf(f64)) return m;
    return m + std.math.log1p(@exp(-@abs(a - b)));
}

/// log(inv_logit(x)) and its derivative, 1 - inv_logit(x), from a single
/// exponential.
pub fn logInvLogitDeriv(x: f64) [2]f64 {
    if (x >= 0) {
        const e = @exp(-x);
        return .{ -std.math.log1p(e), e / (1 + e) };
    }
    const e = @exp(x);
    return .{ x - std.math.log1p(e), 1 / (1 + e) };
}

/// The standard normal quantile function, by Wichura's algorithm AS 241
/// (PPND16), accurate to about 1 part in 10^16.
pub fn normalQuantile(p: f64) f64 {
    if (p <= 0) return -std.math.inf(f64);
    if (p >= 1) return std.math.inf(f64);
    const q = p - 0.5;
    if (@abs(q) <= 0.425) {
        const r = 0.180625 - q * q;
        const num = ((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r +
            45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r +
            133.14166789178437745) * r + 3.387132872796366608;
        const den = ((((((r * 5226.495278852545925 + 28729.085735721942674) * r + 39307.89580009271061) * r +
            21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r +
            42.313330701600911252) * r + 1.0;
        return q * num / den;
    }
    var r = if (q < 0) p else 1 - p;
    r = @sqrt(-@log(r));
    var val: f64 = undefined;
    if (r <= 5) {
        r -= 1.6;
        const num = ((((((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177) * r +
            1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r +
            4.6303378461565452959) * r + 1.42343711074968357734;
        const den = ((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r +
            0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r +
            2.05319162663775882187) * r + 1.0;
        val = num / den;
    } else {
        r -= 5;
        const num = ((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r +
            0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r +
            5.4637849111641143699) * r + 6.6579046435011037772;
        const den = ((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r +
            7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r +
            0.59983220655588793769) * r + 1.0;
        val = num / den;
    }
    return if (q < 0) -val else val;
}

const testing = std.testing;

test "normal quantile matches reference values" {
    // Reference values from scipy.special.ndtri.
    const cases = [_][2]f64{
        .{ 0.5, 0 },
        .{ 0.975, 1.959963984540054 },
        .{ 0.025, -1.9599639845400545 },
        .{ 0.841344746068543, 1.0 },
        .{ 1e-10, -6.361340902404056 },
        .{ 1e-300, -37.0470962993612 },
        .{ 0.999, 3.090232306167813 },
    };
    for (cases) |c| {
        const got = normalQuantile(c[0]);
        try testing.expectApproxEqAbs(c[1], got, 1e-13 * (1 + @abs(c[1])));
    }
}

test "log1pExp and invLogit at extremes" {
    try testing.expectEqual(@as(f64, 800), log1pExp(800));
    try testing.expect(log1pExp(-800) == 0);
    try testing.expectApproxEqAbs(@log(2.0), log1pExp(0), 1e-16);
    try testing.expectEqual(@as(f64, 1), invLogit(800));
    try testing.expectEqual(@as(f64, 0), invLogit(-800));
    try testing.expectApproxEqAbs(1000 + @log(2.0), logSumExp(1000, 1000), 1e-12);
    for ([_]f64{ -40, -2, -1e-3, 0, 1e-3, 2, 40 }) |x| {
        const r = logInvLogitDeriv(x);
        try testing.expectApproxEqRel(-log1pExp(-x), r[0], 1e-14);
        try testing.expectApproxEqRel(invLogit(-x), r[1], 1e-14);
    }
}
