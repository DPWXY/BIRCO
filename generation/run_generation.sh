echo $$ > pid_file.txt

# Define variables
thread_num=5
prompt_path="./generation_data/prompt/arguana_test_set_reason_O.pickle"
name="arguana_test_set_reason_O"

start=0
end=5001

result_prefix= "./generation_data/output/${name}"

while [ $start -lt $end ]; do
   echo "=========================================================================="
   echo "Starting program with start index: ${start}"
   start_time=$(date +%s)

   # Use the variables for file paths and prefixes
   timeout 300 python3 -u gpt_generation.py -o ${result_prefix}_${start}_$((${start}+$thread_num)) -t $thread_num -s ${start} -e $((${start}+$thread_num)) -prompt $prompt_path

   end_time=$(date +%s)
   runtime=$((end_time-start_time))

   echo "Program with generating ${start} to $((${start}+$thread_num)) ran for ${runtime} seconds"

   start=$((${start}+$thread_num))

   echo "Sleeping for 5s"
   sleep 5
done
echo "Finish query generation"

echo "Collecting result"

python3 collect_generated.py -result_dir ./generation_result/generation_output -result_prefix $name -name $name
