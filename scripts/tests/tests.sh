#!/bin/zsh

# cwgangp U2OS 24h
test_result=$(python src/tools/gan/test.py --conf_file confs/cwgangp/trial_wellplate_epoch100_batch28/test_mse_U2OS_24h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# cwgangp A549 24h
test_result=$(python src/tools/gan/test.py --conf_file confs/cwgangp/trial_wellplate_epoch100_batch28/test_mse_A549_24h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# cwgangp A549 48h
test_result=$(python src/tools/gan/test.py --conf_file confs/cwgangp/trial_wellplate_epoch100_batch28/test_mse_A549_48h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# wsb U2OS 24h
test_result=$(python src/tools/custom/wsb/test.py --conf_file confs/wsb/trial_wellplate_epoch100_batch28/test_U2OS_24h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# wsb A549 24h
test_result=$(python src/tools/custom/wsb/test.py --conf_file confs/wsb/trial_wellplate_epoch100_batch28/test_A549_24h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# wsb A549 48h
test_result=$(python src/tools/custom/wsb/test.py --conf_file confs/wsb/trial_wellplate_epoch100_batch28/test_A549_48h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# i2sb U2OS 24h
test_result=$(python src/tools/diffusion/test.py --conf_file confs/i2sb/trial_wellplate_epoch100_batch28/test_U2OS_24h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# i2sb A549 24h
test_result=$(python src/tools/diffusion/test.py --conf_file confs/i2sb/trial_wellplate_epoch100_batch28/test_A549_24h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# i2sb A549 48h
test_result=$(python src/tools/diffusion/test.py --conf_file confs/i2sb/trial_wellplate_epoch100_batch28/test_A549_48h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# palette U2OS 24h
test_result=$(python src/tools/diffusion/test.py --conf_file confs/palette/trial_wellplate_epoch100_batch28/test_U2OS_24h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# palette A549 24h
test_result=$(python src/tools/diffusion/test.py --conf_file confs/palette/trial_wellplate_epoch100_batch28/test_A549_24h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# palette A549 48h
test_result=$(python src/tools/diffusion/test.py --conf_file confs/palette/trial_wellplate_epoch100_batch28/test_A549_48h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# guidedI2I U2OS 24h
test_result=$(python src/tools/diffusion/test.py --conf_file confs/guidedI2I/trial_wellplate_epoch100_batch28/test_U2OS_24h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# guidedI2I A549 24h
test_result=$(python src/tools/diffusion/test.py --conf_file confs/guidedI2I/trial_wellplate_epoch100_batch28/test_A549_24h.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"

# guidedI2I A549 48h
test_result=$(python src/tools/diffusion/test.py --conf_file confs/guidedI2I/trial_wellplate_epoch100_batch28/test_A549_48h.cfg| tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"