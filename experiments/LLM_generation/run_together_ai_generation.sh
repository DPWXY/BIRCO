echo $$ > pid_file.txt

# Define variables
thread_num=25
model_name="llama_13b"
model_path="togethercomputer/llama-2-13b-chat"
prompt_name="arguana_test_set_score_O"

echo "LM model:" 
echo $model_name
echo $model_path

echo "Start generation"

prompt_path="prompts/llama_stripedhyena/${prompt_name}.pickle"
result_prefix="${model_name}_${prompt_name}"
name="${model_name}_${prompt_name}"

start=0
end=100

while [ $start -lt $end ]; do
   echo "=========================================================================="
   echo "Starting program with start index: ${start}"
   start_time=$(date +%s)

   # Use the variables for file paths and prefixes
   timeout 120 python3 -u together_ai.py -o ${result_prefix}_${start}_$((${start}+$thread_num)) -t $thread_num -s ${start} -e $((${start}+$thread_num)) -prompt $prompt_path -model_name $model_path
   end_time=$(date +%s)
   runtime=$((end_time-start_time))
   echo "Program with generating ${start} to $((${start}+$thread_num)) ran for ${runtime} seconds"
   start=$((${start}+$thread_num))

done
echo "Finish query generation"
echo "Collecting result"
python3 collect_generated.py -result_dir ./generation_result/ -result_prefix $name -name $name
echo "=========================================================================="
