//! Forward-mode automatic differentiation with SIMD tangents.
//!
//! A `Dual(n)` holds a value and its gradient with respect to `n` inputs. The
//! gradient is a `@Vector(n, f64)`, so every operation updates all `n` partial
//! derivatives at once with vector instructions. Evaluating a log density on
//! `n` dual inputs gives its value and full gradient in a single pass.
//!
//! The cost of one evaluation is the number of operations times `n`, so this
//! suits models with up to a few hundred parameters. Reverse mode (what Stan
//! uses) costs a constant multiple of one evaluation regardless of `n`.

const std = @import("std");
const smath = @import("math.zig");

pub fn Dual(comptime n: usize) type {
    return struct {
        const Self = @This();
        pub const dim = n;
        pub const Vec = @Vector(n, f64);

        val: f64,
        grad: Vec,

        inline fn sp(x: f64) Vec {
            return @splat(x);
        }

        /// A constant: zero gradient.
        pub inline fn constant(x: f64) Self {
            return .{ .val = x, .grad = @splat(0) };
        }

        /// The i-th input: value x, gradient the i-th unit vector.
        pub fn variable(x: f64, i: usize) Self {
            var g: [n]f64 = @splat(0);
            g[i] = 1;
            return .{ .val = x, .grad = g };
        }

        /// Seeds an array of inputs from plain values.
        pub fn variables(x: *const [n]f64) [n]Self {
            var out: [n]Self = undefined;
            for (&out, x, 0..) |*o, xi, i| o.* = variable(xi, i);
            return out;
        }

        /// The value with a derivative `d` applied through the chain rule.
        pub inline fn chain(a: Self, value: f64, d: f64) Self {
            return .{ .val = value, .grad = a.grad * sp(d) };
        }

        pub inline fn add(a: Self, b: Self) Self {
            return .{ .val = a.val + b.val, .grad = a.grad + b.grad };
        }

        pub inline fn sub(a: Self, b: Self) Self {
            return .{ .val = a.val - b.val, .grad = a.grad - b.grad };
        }

        pub inline fn neg(a: Self) Self {
            return .{ .val = -a.val, .grad = -a.grad };
        }

        pub inline fn mul(a: Self, b: Self) Self {
            return .{ .val = a.val * b.val, .grad = a.grad * sp(b.val) + b.grad * sp(a.val) };
        }

        pub inline fn div(a: Self, b: Self) Self {
            const q = a.val / b.val;
            return .{ .val = q, .grad = (a.grad - b.grad * sp(q)) / sp(b.val) };
        }

        /// a + c for a constant c.
        pub inline fn addC(a: Self, c: f64) Self {
            return .{ .val = a.val + c, .grad = a.grad };
        }

        /// c - a for a constant c.
        pub inline fn rsubC(a: Self, c: f64) Self {
            return .{ .val = c - a.val, .grad = -a.grad };
        }

        /// a * c for a constant c.
        pub inline fn scale(a: Self, c: f64) Self {
            return .{ .val = a.val * c, .grad = a.grad * sp(c) };
        }

        /// a * c + b for a constant c.
        pub inline fn mulAdd(a: Self, c: f64, b: Self) Self {
            return .{ .val = @mulAdd(f64, a.val, c, b.val), .grad = a.grad * sp(c) + b.grad };
        }

        /// 1 / a.
        pub inline fn recip(a: Self) Self {
            const r = 1 / a.val;
            return chain(a, r, -r * r);
        }

        pub inline fn square(a: Self) Self {
            return chain(a, a.val * a.val, 2 * a.val);
        }

        pub inline fn exp(a: Self) Self {
            const e = @exp(a.val);
            return chain(a, e, e);
        }

        pub inline fn log(a: Self) Self {
            return chain(a, @log(a.val), 1 / a.val);
        }

        pub inline fn log1p(a: Self) Self {
            return chain(a, std.math.log1p(a.val), 1 / (1 + a.val));
        }

        pub inline fn sqrt(a: Self) Self {
            const s = @sqrt(a.val);
            return chain(a, s, 0.5 / s);
        }

        /// log(1 + exp(a)), the softplus function.
        pub inline fn log1pExp(a: Self) Self {
            return chain(a, smath.log1pExp(a.val), smath.invLogit(a.val));
        }

        /// 1 / (1 + exp(-a)).
        pub inline fn invLogit(a: Self) Self {
            const s = smath.invLogit(a.val);
            return chain(a, s, s * (1 - s));
        }

        /// log(inv_logit(a)) = -log1p(exp(-a)).
        pub inline fn logInvLogit(a: Self) Self {
            const r = smath.logInvLogitDeriv(a.val);
            return chain(a, r[0], r[1]);
        }

        /// log(1 - inv_logit(a)) = log(inv_logit(-a)).
        pub inline fn log1mInvLogit(a: Self) Self {
            const r = smath.logInvLogitDeriv(-a.val);
            return chain(a, r[0], -r[1]);
        }

        /// log(exp(a) + exp(b)) without overflow.
        pub inline fn logSumExp(a: Self, b: Self) Self {
            const hi = if (a.val >= b.val) a else b;
            const lo = if (a.val >= b.val) b else a;
            if (hi.val == -std.math.inf(f64)) return a;
            const e = @exp(lo.val - hi.val);
            const w = 1 / (1 + e);
            return .{ .val = hi.val + std.math.log1p(e), .grad = hi.grad * sp(w) + lo.grad * sp(e * w) };
        }

        /// The sum of an array.
        pub fn sum(xs: []const Self) Self {
            var acc = constant(0);
            for (xs) |x| acc = acc.add(x);
            return acc;
        }
    };
}

/// A plain value with the same interface as `Dual`, so that a model written
/// once can be evaluated without derivatives (for output and for checks).
pub const Real = struct {
    const Self = @This();

    val: f64,

    pub inline fn constant(x: f64) Self {
        return .{ .val = x };
    }
    pub inline fn add(a: Self, b: Self) Self {
        return .{ .val = a.val + b.val };
    }
    pub inline fn sub(a: Self, b: Self) Self {
        return .{ .val = a.val - b.val };
    }
    pub inline fn neg(a: Self) Self {
        return .{ .val = -a.val };
    }
    pub inline fn mul(a: Self, b: Self) Self {
        return .{ .val = a.val * b.val };
    }
    pub inline fn div(a: Self, b: Self) Self {
        return .{ .val = a.val / b.val };
    }
    pub inline fn addC(a: Self, c: f64) Self {
        return .{ .val = a.val + c };
    }
    pub inline fn rsubC(a: Self, c: f64) Self {
        return .{ .val = c - a.val };
    }
    pub inline fn scale(a: Self, c: f64) Self {
        return .{ .val = a.val * c };
    }
    pub inline fn mulAdd(a: Self, c: f64, b: Self) Self {
        return .{ .val = @mulAdd(f64, a.val, c, b.val) };
    }
    pub inline fn recip(a: Self) Self {
        return .{ .val = 1 / a.val };
    }
    pub inline fn square(a: Self) Self {
        return .{ .val = a.val * a.val };
    }
    pub inline fn exp(a: Self) Self {
        return .{ .val = @exp(a.val) };
    }
    pub inline fn log(a: Self) Self {
        return .{ .val = @log(a.val) };
    }
    pub inline fn log1p(a: Self) Self {
        return .{ .val = std.math.log1p(a.val) };
    }
    pub inline fn sqrt(a: Self) Self {
        return .{ .val = @sqrt(a.val) };
    }
    pub inline fn log1pExp(a: Self) Self {
        return .{ .val = smath.log1pExp(a.val) };
    }
    pub inline fn invLogit(a: Self) Self {
        return .{ .val = smath.invLogit(a.val) };
    }
    pub inline fn logInvLogit(a: Self) Self {
        return .{ .val = smath.logInvLogitDeriv(a.val)[0] };
    }
    pub inline fn log1mInvLogit(a: Self) Self {
        return .{ .val = smath.logInvLogitDeriv(-a.val)[0] };
    }
    pub inline fn logSumExp(a: Self, b: Self) Self {
        return .{ .val = smath.logSumExp(a.val, b.val) };
    }
    pub fn sum(xs: []const Self) Self {
        var acc: f64 = 0;
        for (xs) |x| acc += x.val;
        return .{ .val = acc };
    }
};

const testing = std.testing;

fn expectClose(a: f64, b: f64, tol: f64) !void {
    if (!(@abs(a - b) <= tol * (1 + @abs(b)))) {
        std.debug.print("expected {d}, got {d}\n", .{ b, a });
        return error.TestExpectedApproxEq;
    }
}

test "dual arithmetic matches analytic derivatives" {
    const D = Dual(2);
    const x = D.variable(0.7, 0);
    const y = D.variable(-1.3, 1);
    // f = x * y / (1 + x^2) + exp(y) * log(x + 2)
    const f = x.mul(y).div(x.square().addC(1)).add(y.exp().mul(x.addC(2).log()));
    const fx = 0.7;
    const fy = -1.3;
    const value = fx * fy / (1 + fx * fx) + @exp(fy) * @log(fx + 2);
    const dfx = fy * (1 - fx * fx) / ((1 + fx * fx) * (1 + fx * fx)) + @exp(fy) / (fx + 2);
    const dfy = fx / (1 + fx * fx) + @exp(fy) * @log(fx + 2);
    try expectClose(f.val, value, 1e-15);
    try expectClose(f.grad[0], dfx, 1e-14);
    try expectClose(f.grad[1], dfy, 1e-14);
}

test "logistic helpers are stable and consistent" {
    const D = Dual(1);
    for ([_]f64{ -800, -30, -1, 0, 0.5, 30, 800 }) |v| {
        const a = D.variable(v, 0);
        const s = smath.invLogit(v);
        try testing.expect(std.math.isFinite(a.log1pExp().val));
        try testing.expect(std.math.isFinite(a.logInvLogit().val));
        try testing.expect(std.math.isFinite(a.log1mInvLogit().val));
        try expectClose(a.invLogit().grad[0], s * (1 - s), 1e-15);
        try expectClose(a.logInvLogit().grad[0], 1 - s, 1e-15);
        try expectClose(a.log1mInvLogit().grad[0], -s, 1e-15);
        // log(s) + log(1 - s) = log(s (1 - s))
        if (@abs(v) < 30) try expectClose(a.logInvLogit().val + a.log1mInvLogit().val, @log(s * (1 - s)), 1e-12);
    }
}

test "logSumExp gradient is the softmax" {
    const D = Dual(2);
    const a = D.variable(1000, 0);
    const b = D.variable(999, 1);
    const r = a.logSumExp(b);
    const wa = 1.0 / (1.0 + @exp(-1.0));
    try expectClose(r.val, 1000 + @log(1 + @exp(-1.0)), 1e-15);
    try expectClose(r.grad[0], wa, 1e-15);
    try expectClose(r.grad[1], 1 - wa, 1e-14);
}
