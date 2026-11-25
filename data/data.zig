//! The posteriordb data sets used by the models, embedded in the binary.
//! The files are copied unchanged from posteriordb (BSD 3-Clause License,
//! see LICENSE in this directory).

const std = @import("std");

const files = .{
    .{ "eight_schools.json", @embedFile("posteriordb/eight_schools.json") },
    .{ "arK.json", @embedFile("posteriordb/arK.json") },
    .{ "gp_pois_regr.json", @embedFile("posteriordb/gp_pois_regr.json") },
    .{ "low_dim_gauss_mix.json", @embedFile("posteriordb/low_dim_gauss_mix.json") },
    .{ "wells_data.json", @embedFile("posteriordb/wells_data.json") },
    .{ "garch.json", @embedFile("posteriordb/garch.json") },
    .{ "dogs.json", @embedFile("posteriordb/dogs.json") },
};

/// The contents of an embedded data file, or null if there is none by that name.
pub fn get(name: []const u8) ?[]const u8 {
    inline for (files) |f| {
        if (std.mem.eql(u8, f[0], name)) return f[1];
    }
    return null;
}
