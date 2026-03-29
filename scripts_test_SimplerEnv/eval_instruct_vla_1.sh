# ========================================
# SimplerEnv 多任务并行评测脚本（InstructVLA）
# 用法：仅需修改 ckpt_path 为待评测模型 checkpoint。
# ========================================

# 只需要替换这里的 checkpoint 路径。
ckpt_path="TBD"

# 进入 SimplerEnv 根目录，后续脚本路径均按该目录解析。
cd ./SimplerEnv

# 结果目录规则：以 checkpoint 上两级目录为基路径，按文件名创建结果子目录。
base_dir=$(dirname "$ckpt_path")
base_dir=$(dirname "$base_dir")
file_name=$(basename "$ckpt_path" .pt)
result_path="${base_dir}/results_${file_name}"

# 兼容部分环境中的 cuDNN 动态库加载问题。
export LD_LIBRARY_PATH=~/miniconda3/envs/instructvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=~/miniconda3/envs/instructvla/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 创建评测结果目录。
if [ ! -d "$result_path" ]; then
    mkdir -p "$result_path"
fi

# 创建日志目录，用于保存每个子任务的 stdout/stderr。
log_path="${result_path}/log/"
if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi


# 显式注入 PYTHONPATH，确保自定义 policy 能被正确 import。
export PYTHONPATH="/mnt/petrelfs/yangshuai1/rep/InstructVLA_official/SimplerEnv:$PYTHONPATH" 

# 以下 8 个脚本并行启动，覆盖不同子任务切分。
# 每个脚本第二个参数为分片编号（0~7）。
bash ./scripts_self/situated_1.sh $ckpt_path 0 > "${log_path}/log1.log" 2>&1 &
pid1=$!
echo "1: $pid1"

bash ./scripts_self/situated_2.sh $ckpt_path 1 > "${log_path}/log2.log" 2>&1 &
pid2=$!
echo "2: $pid2"

bash ./scripts_self/situated_3.sh $ckpt_path 2 > "${log_path}/log3.log" 2>&1 &
pid3=$!
echo "3: $pid3"

bash ./scripts_self/aggregation_1.sh $ckpt_path 3 > "${log_path}/log4.log" 2>&1 &
pid4=$!
echo "4: $pid4"

bash ./scripts_self/aggregation_2.sh $ckpt_path 4 > "${log_path}/log5.log" 2>&1 &
pid5=$!
echo "5: $pid5"

bash ./scripts_self/aggregation_3.sh $ckpt_path 5 > "${log_path}/log6.log" 2>&1 &
pid6=$!
echo "6: $pid6"

bash ./scripts_self/aggregation_4.sh $ckpt_path 6 > "${log_path}/log7.log" 2>&1 &
pid7=$!
echo "7: $pid7"

bash ./scripts_self/aggregation_5.sh $ckpt_path 7 > "${log_path}/log8.log" 2>&1 &
pid8=$!
echo "8: $pid8"

# 阻塞等待全部子进程完成。
wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8

echo "Done!"

echo "computing final_results_instruct.log ... "

# 结果聚合脚本读取的目录是 checkpoint 同名目录（由各子脚本生成）。
full_path="${result_path}/$(basename "$ckpt_path")"

# 汇总 instruction-following 评测指标，输出到最终日志。
python calc_instruction_folliowing.py --results-dir $full_path  > "${log_path}/final_results_instruct.log" 2>&1