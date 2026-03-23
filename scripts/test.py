from datasets import load_dataset

dset = load_dataset("jmhessel/newyorker_caption_contest", "ranking")
print(dset.column_names)