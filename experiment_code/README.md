## Running Guideline

Below is a step-by-step guideline for reproducing the reported experimental results (Note: The main experimental codes are based on and modified from https://github.com/lorenzkuhn/semantic_uncertainty):

1. Data preparation: 
- (a) set the paths for huggingface model and dataset cache in ```config.py```. 
- (b) download the CoQA dataset from https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json, and place it in the ```{data_dir}``` specified in ```config.py```. 
- (c) run command ```python parse_coqa.py``` to parse CoQA dataset. 
- (d) run command ```python parse_triviaqa.py --model={model_name}``` to download and parse TriviaQA dataset. 
- (e) download SciQ dataset form https://github.com/launchnlp/LitCab/blob/main/sciq/test.txt and put ```test.txt``` in ```{config.data_dir}/sciq```. 
- (f) download NQ dataset from https://github.com/launchnlp/LitCab/blob/main/NQ/test.txt and put ```test.txt``` in ```{config.data_dir}/NQ```. 
- (g) run command ```python parse_datasets.py --model={model_name} --dataset={dataset_name}``` for parsing SciQ and NQ dataset. 

2. Generate responses: 
- (a) run command ```python generate_beam_search_save_all_triviaqa_coqa_cleaned_device.py --num_generations_per_prompt='10' --model={model_name} --fraction_of_data_to_use='0.2'--num_beams='10' --top_p='1.0' --dataset='coqa' --cuda_device={cuda_device_id}``` to generate responses for CoQA dataset. 
- (b) run command ```python generate_beam_search_save_all_triviaqa_coqa_cleaned_device.py --num_generations_per_prompt='10' --model={model_name} --fraction_of_data_to_use='0.1'--num_beams='10' --top_p='1.0' --dataset='trivia_qa' --cuda_device={cuda_device_id}``` to generate responses for TriviaQA dataset. 
- (c) run command ```python generate_beam_search_save_all_datasets_cleaned_device.py --num_generations_per_prompt='10' --model={model_name} --fraction_of_data_to_use='1.0'--num_beams='10' --top_p='1.0' --dataset='sciq' --cuda_device={cuda_device_id}``` to generate responses for SciQ dataset. 
- (d) run command ```python generate_beam_search_save_all_datasets_cleaned_device.py --num_generations_per_prompt='10' --model={model_name} --fraction_of_data_to_use='0.5'--num_beams='10' --top_p='1.0' --dataset='NQ' --cuda_device={cuda_device_id}``` to generate responses for NQ dataset.

3. Calculate pair-wise semantic similarities for semantic entropy: 
- run command ```python get_semantic_similarities_beam_search_datasets.py --generation_model={model_name} --dataset={dataset_name}```  

4. Calculate likelihood information: 
- (a) run command ```python get_likelihoods_beam_search_datasets_temperature.py --evaluation_model={model_name} --generation_model={model_name} --dataset={dataset_name} --cuda_device={cuda_device_id} --temperature=0.1``` 
- (b) run command ```python get_likelihoods_beam_search_datasets_temperature.py --evaluation_model={model_name} --generation_model={model_name} --dataset={dataset_name} --cuda_device={cuda_device_id} --temperature=1.0```

5. Calculate rouge scores: 
- run command ```python calculate_beam_search_rouge_datasets.py --model={model_name} --dataset={dataset_name}```

6. Calculate P(True): 
- run command ```python get_prompting_based_uncertainty_beam_search.py --generation_model={model_name} --dataset={dataset_name} --cuda_device={cuda_device_id}``` 

7. Calculate semantic density: 
- (a) run command ```python get_semantic_density_full_beam_search_unique_datasets_temperature.py --generation_model={model_name} --dataset={dataset_name} --cuda_device={cuda_device_id} --temperature=0.1``` 
- (b) run command ```python get_semantic_density_full_beam_search_unique_datasets_temperature.py --generation_model={model_name} --dataset={dataset_name} --cuda_device={cuda_device_id} --temperature=1.0```

8. Calculate semantic density with different numbers of reference responses: 
- (a) run command ```python get_semantic_density_full_beam_search_unique_datasets_temperature_sample_num.py --generation_model={model_name} --dataset={dataset_name} --cuda_device={cuda_device_id} --temperature=0.1``` 
- (b) run command ```python get_semantic_density_full_beam_search_unique_datasets_temperature_sample_num.py --generation_model={model_name} --dataset={dataset_name} --cuda_device={cuda_device_id} --temperature=1.0```

9. Calculate AUROC scores for all the uncertainty metrics: 
- (a) run command ```python compute_confidence_measure_beam_search_unique_temperature.py --generation_model={model_name} --evaluation_model={model_name} --dataset={dataset_name} --temperature=0.1 --cuda_device={cuda_device_id}``` and command ```python compute_confidence_measure_beam_search_unique_temperature.py --generation_model={model_name} --evaluation_model={model_name} --dataset={dataset_name} --temperature=1.0 --cuda_device={cuda_device_id}``` 
- (b) create a fold named ```results``` to store auroc results. 
- (c) run command ```python analyze_results_semantic_density_full_datasets_temperature.py --dataset={dataset_name} --model={model_name} --temperature=0.1 --cuda_device={cuda_device_id}``` and command ```python analyze_results_semantic_density_full_datasets_temperature.py --dataset={dataset_name} --model={model_name} --temperature=1.0 --cuda_device={cuda_device_id}``` 
- (d) run command ```python analyze_results_semantic_density_full_datasets_temperature_sample_num.py  --dataset={dataset_name} --model={model_name} --temperature=0.1 --cuda_device={cuda_device_id} --sample_num=10```

10. Generate results shown in the paper: 
- (a) create a folder named ```paper_results``` to store table results and a folder named ```plots``` to save the figures. 
- (b) run command ```python results_table_auroc.py --dataset={dataset_name} --temperature=0.1``` to generate the results in Table 1. 
- (c) run command ```python results_table_auroc_statistical_test.py --temperature=0.1``` to generate the results in Table A1. 
- (d) run command ```python results_sample_num_auroc.py --dataset={dataset_name} --temperature=0.1 --sample_num=10``` and command ```python plot_sample_num_auroc.py --dataset={dataset_name} --temperature=0.1 --sample_num=10``` to generate the plots in Figure 1. 
- (e) run command ```python results_group_auroc_average_over_datasets.py --temperature=0.1``` and command ```python plot_group_auroc_average_over_datasets.py --temperature=0.1``` to generate Figure 2. 
