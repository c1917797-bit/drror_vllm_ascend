# Project review of the registered AscendC operator template

Reviewed 2026-09-09 by the main agent before implementation. This is a
project-local adaptation record, not a replacement shared skill and not a
claim that a native prototype has been built or passed.

## Read sources and applicability

Read in full under the read-only reference root
`/cache/cch/cannbot-skills/ops/ascendc-registry-invoke-template`:

- `SKILL.md`;
- `references/basic-guide.md`, `advanced-guide.md`, `build-deploy-guide.md`;
- `references/add_example/build.sh`;
- the example root, op_host and op_kernel CMakeLists;
- `references/add_example/op_host/add_example_def.cpp`.

The registered OpDef/tiling/kernel/ACLNN structure is relevant to an
independently named recurrent control and later decode layout/fusion work.
The example is an elementwise Add, not a stateful GDN implementation.
Template completion, CPU tests and successful compilation would not prove
NPU numerical correctness, graph correctness or end-to-end benefit.

The actual dedicated container uses CANN 9.1.0. Its
`aarch64-linux/lib64/cmake/asc-config.cmake` and the ASC framework
functions exist. This verifies API availability, not that a new project
can already build. Use the image-pinned frozen recurrent implementation
as the semantic control; the current ops-transformer reference has a
different BF16-only state API and must not be substituted silently.

## Demonstrated mismatches and project corrections

1. **Installation side effects.** The example build script runs the
   generated installer automatically for `-s`, `-e` and thus `-a`.
   Do not run those commands or the installer against the original image,
   global CANN tree, existing vendor or reference checkout. Build/package
   and explicit private loading are separate actions. A build-only
   command must be reviewed for its actual output paths.

2. **Cleanup and output scope.** The example removes a CMake cache during
   configuration and has recursive wildcard cleanup of build directories.
   Our entry must use a fresh explicitly resolved plugin-owned build
   directory, reject pre-existing outputs, and provide no implicit clean,
   overwrite, install or automatic retry. Source, build recipe and artifact
   identities stay in this plugin repository. A dedicated build container
   may expose only the required plugin build subtree writable and no NPU
   devices; do not change the serving container's read-only source mount.

3. **Architecture assumptions.** The text and template mix the name
   ARCH32 with arch22 directory paths, and their default CMake selects
   multiple chips. The template OpDef actually registers ascend910b and
   ascend950, although its shell also lists ascend910_93. Our first
   prototype targets only verified 910B4 / ascend910b. No arch35 or
   multi-chip support is inferred from template names.

4. **State mutation and layout.** The Add OpDef marks inputs/output
   AutoContiguous and the generic L2 recipe calls Contiguous/ViewCopy.
   These are not an admitted contract for recurrent in-place state or a
   packed strided reader. Preserve the frozen input/output alias
   semantics and torch `Tensor(a!)` mutation declaration. Validate and
   reject unsupported shapes, dtypes and strides; do not hide a copy.
   First establish the unchanged contiguous native ABI under new names
   before adding an explicit packed-value reader.

5. **Namespace and dispatch.** The template's standard library basename
   and package generation are not runtime isolation. Use a unique vendor,
   OpDef/tiling/kernel/ACLNN names and torch namespace. Resolve the new
   ACLNN symbols from an explicit absolute path and check their actual
   dladdr owners. Never overwrite `_C_ascend` registrations or the
   custom_transformer vendor. Mapping a library is not proof that its
   function or device binary was selected.

6. **Numerical claims.** BF16 and FP32 recurrent state require their own
   NPU coverage. The frozen arithmetic and BF16 rounding boundaries are
   retained in the copy/control stage. Pure storage changes require
   exact finite logical tensor bytes, complete multi-step state, guards,
   unselected slots, metadata stability and NPUGraph replay. Later
   arithmetic fusion needs justified predeclared incremental tolerances,
   native and oracle comparisons; no tolerance change after a failure.

## Action gates

Before a build: pin source and license provenance, inspect the minimum
support files, compiler/CANN/torch ABI, explicit output and package paths,
and the generated API's state mutation behavior. Preserve source notices.

Before an NPU run: follow AGENTS.md admission rules and recheck physical
4-7 isolation. One bounded representative matrix, no automatic retries.
The ordinary service wrapper and paired throughput protocol are used only
after numerical and graph admission. Common optimizations apply to dense
as well. Historical A/B and A/A logprob failures remain failed; this skill
does not authorize the deferred framework determinism experiment.

Final acceptance remains three matched performance rounds at fixed 1024
output tokens and thinking-off, normal-EOS, max_tokens=2048 paired
LongBench-v2 and GSM8K. This review does not report any new performance or
task-accuracy result.

