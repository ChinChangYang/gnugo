# GNUGo

## Building

```sh
./configure
make -j$(sysctl -n hw.logicalcpu)
```

The binary is built at `interface/gnugo`.

## Running Regression Tests

Run the `first_batch` (13 test files) from the `regression/` directory:

```sh
cd regression && for tst in reading.tst owl.tst ld_owl.tst optics.tst filllib.tst atari_atari.tst connection.tst break_in.tst blunder.tst unconditional.tst trevora.tst nngs1.tst strategy.tst; do echo "$tst: $(../interface/gnugo --quiet --mode gtp < $tst 2>&1 | awk -f regress.awk tst=$tst verbose=1 2>&1 | grep '^Summary:')"; done
```

The awk script only prints unexpected failures/passes by default. Pass `verbose=1` to see all results including the summary line.

## Debugging Failed Test Cases

Before tracing why a test fails, first retry at a higher level (`--level 12`) to check if the failure is a node/depth limit artifact (the most common cause). If it passes at higher level, the bug is budget starvation, not a logic error.
