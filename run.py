import os
import sys
import json

def main(targets):

	# print(targets)

	if 'data' in targets:
		with open('config/data-params.json') as fh:
			data_cfg = json.load(fh)
			
		# make the data target
		data = get_data(**data_cfg)

	if 'features' in targets:
		with open('config/features-params.json') as fh:
			feats_cfg = json.load(fh)
		
		feats, labels = apply_features(data, **feats_cfg)

	if 'model' in targets:
		with open('config/model-params.json') as fh:
			model_cfg = json.load(fh)

		# make the data target
		model_build(feats, labels, **model_cfg)

	if "test" in targets:
		if not os.path.isdir('models'):
			print("No models available.")
			sys.exit(0)

		if not os.path.isdir('results/model_prediction'):
			os.mkdir('results/model_prediction')

		data = targets[1]

		os.system("python3 models/lda_model.py " + data)
		os.system("python3 models/dashboard.py")


if __name__ == '__main__':
    if not os.path.isdir('results'):
        os.makedirs('results')
    targets = sys.argv[1:]
    main(targets)