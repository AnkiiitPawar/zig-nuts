const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const data = b.createModule(.{
        .root_source_file = b.path("data/data.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mod = b.addModule("nuts", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "pdb_data", .module = data }},
    });

    const exe = b.addExecutable(.{
        .name = "znuts",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "nuts", .module = mod },
                .{ .name = "pdb_data", .module = data },
            },
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run znuts").dependOn(&run_cmd.step);

    const tests = b.addTest(.{ .root_module = mod });
    b.step("test", "Run the unit tests").dependOn(&b.addRunArtifact(tests).step);
}
