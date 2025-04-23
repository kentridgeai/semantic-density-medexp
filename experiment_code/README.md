## Running Guideline

Below is a step-by-step guideline for reproducing the reported experimental results (Note: The main experimental codes are based on and modified from https://github.com/cognizant-ai-labs/semantic-density-paper, which in turn is based on and modified from https://github.com/lorenzkuhn/semantic_uncertainty):

1. Data preparation: 
- (a) set the paths for huggingface model and dataset cache in ```config.py```. 
- (b) download the PubMedQA dataset from https://github.com/pubmedqa/pubmedqa/blob/master/data/ori_pqal.json, and place it in the ```{data_dir}``` specified in ```config.py```.
- (c) run command ```python split_pubmedqa.py pqal``` followed by ```python parse_pubmedqa.py``` to parse PubMedQA dataset.
- (d) navigate to the parent folder of this git directory and download the MedExQA dataset via ```git clone https://huggingface.co/datasets/bluesky333/MedExQA``` (i.e. {parent}/semantic-density-medexp, {parent}/MedExQA)
- (d) run command ```python parse_medexqa.py``` to parse MedExQA dataset. 

2. Generate responses: 
- (a) run command ```python generate_beam_search_save_all_triviaqa_coqa_cleaned_device.py --num_generations_per_prompt='10' --model={model_name} --fraction_of_data_to_use='1.0'--num_beams='10' --top_p='1.0' --dataset='pubmedqa' --cuda_device={cuda_device_id}``` to generate responses for PubMedQA dataset. 
- (b) run command ```python generate_beam_search_save_all_medexqa_cleaned_device.py --num_generations_per_prompt='10' --model={model_name} --fraction_of_data_to_use='1.0' --num_beams='10' --top_p='1.0' --dataset='medexqa' --cuda_device={cuda_device_id``` to generate responses for MedExQA dataset.

3. Calculate pair-wise semantic similarities for semantic entropy: 
- run command ```python get_ss_BATCHED_beam_search_datasets.py --generation_model={model_name} --dataset={dataset_name} --device={cuda_device_id}```  

4. Calculate likelihood information: 
- (a) run command ```python get_likelihoods_beam_search_datasets_temperature.py --model={model_name} --dataset={dataset_name} --cuda_device={cuda_device_id} --temperature=0.1```

5. Calculate rouge scores: 
- run command ```python calculate_beam_search_rouge_datasets.py --model={model_name} --dataset={dataset_name} --device={cuda_device_id}```

6. Calculate P(True): 
- run command ```python get_prompting_based_uncertainty_beam_search.py --model={model_name} --dataset={dataset_name} --cuda_device={cuda_device_id}``` 

7. Calculate semantic density: 
- (a) run command ```python get_semantic_density_full_beam_search_unique_datasets_temperature.py --generation_model={model_name} --dataset={dataset_name} --cuda_device={cuda_device_id} --temperature=0.1```

8. Calculate AUROC scores for all the uncertainty metrics: 
- (a) run command ```python compute_confidence_measure_beam_search_unique_temperature.py --generation_model={model_name} --evaluation_model={model_name} --dataset={dataset_name} --temperature=0.1 --cuda_device={cuda_device_id}```
- (b) create a folder named ```results``` to store auroc results.
- (c) run command ```python analyze_results_semantic_density_full_datasets_temperature.py --dataset={dataset_name} --model={model_name} --temperature=0.1 --cuda_device={cuda_device_id}```

10. Generate results shown in the paper: 
- (a) create a folder named ```paper_results``` to store table results and a folder named ```plots``` to save the figures. 
- (b) run command ```python results_table_auroc.py --dataset={dataset_name} --temperature=0.1``` to generate the results in Table 1. 
- (c) run command ```python tables_distribution.py --model={model_name} --dataset={dataset_name} --temperature=0.1 --cuda_device={cuda_device_id}``` to generate the results in Table 2. 
- (d) run command ```python extract_SD_RougeL.py --model={model_name} --dataset={dataset_name} --temperature=0.1 --cuda_device={cuda_device_id}``` and command ```python results_roc_curves.py --dataset={dataset_name} --sample_num=10``` to generate the plots in Figure 1.