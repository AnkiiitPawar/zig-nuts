//! A No-U-Turn sampler in Zig: Stan's multinomial NUTS with its step-size and
//! diagonal metric adaptation, gradients from forward-mode dual numbers on SIMD
//! vectors, and rank-normalised convergence diagnostics.

pub const dual = @import("dual.zig");
pub const math = @import("math.zig");
pub const models = @import("models.zig");
pub const nuts = @import("nuts.zig");
pub const diagnostics = @import("diagnostics.zig");

test {
    _ = dual;
    _ = math;
    _ = models;
    _ = nuts;
    _ = diagnostics;
}
