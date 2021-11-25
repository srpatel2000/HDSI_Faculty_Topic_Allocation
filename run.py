import os
import sys
import json


def main(targets):

	# print(targets)

	if "test" in targets:
		if not os.path.isdir('models'):
			print("No models available.")
			sys.exit(0)

		if not os.path.isdir('results/model_prediction'):
			os.mkdir('results/model_prediction')

		data = targets[1]

		os.system("python3 models/lda_model.py" + data)


if __name__ == '__main__':
    if not os.path.isdir('results'):
        os.makedirs('results')
    targets = sys.argv[1:]
    main(targets)