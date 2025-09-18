python ./convert/chatml_converter.py ./output/data/1758011355/dpo.json --convert-type dpo --reject-count 2 --output-dir ./data/train \
&& python ./convert/chatml_converter.py ./output/data/1758011355/dpo.json --convert-type sft --output-dir ./data/train  \
&& python ./convert/chatml_converter.py ./output/data/1758011355/full_docs.json --convert-type cpt --output-dir ./data/train
