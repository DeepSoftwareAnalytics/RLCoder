import pandas as pd
import random

class CodeBlock(object):
    def __init__(self, file_path, description, code_content, language, _type):
        """
        Represents a block of code.
        :param file_path: The path to the code file.
        :param description: The description of the code block.
        :param code_content: The content of the code block.
        """
        self.file_path = file_path
        self.code_content = code_content
        self.description = description
        self.language = language
        self._type = _type

    def __str__(self):
        if self.language == "python":
            comment_label = "#"
        else:
            comment_label = "//"
        crossfile_context = "\n".join([f"{comment_label} {cl}" for cl in  self.description.strip().split('\n') if cl]) + "\n"
        crossfile_context += "\n".join([f"{comment_label} {cl}" for cl in  self.code_content.split('\n') if cl])
        return crossfile_context.strip()

class Example(object):
    def __init__(self, task_id, file_path, left_context, right_context, related_files, target_code, language):
        """
        Represents an example used for constructing a dataset.
        :param task_id: Task ID.
        :param file_path: File path.
        :param left_context: The context to the left of the target code.
        :param right_context: The context to the right of the target code.
        :param related_files: A list of related files, each containing a path and text.
        :param target_code: The target code snippet.
        """
        self.task_id = task_id
        self.file_path = file_path
        self.left_context = left_context
        self.right_context = right_context
        self.related_files = related_files
        self.target_code = target_code
        self.language = language

    def __str__(self):
        return (f"[Example]:\n"
                f"[Task ID]:\n{self.task_id}\n"
                f"[Path]:\n{self.file_path}\n"
                f"[Left Context]:\n{self.left_context}\n"
                f"[Target Code]:\n{self.target_code}\n"
                f"[Right Context]:\n{self.right_context}\n"
                f"[Related Files]:\n{len(self.related_files)} files\n"
        )
    
def load_test_dataset(args, datasetname, language):
    """
    Loads a dataset.
    :param args: Parameters containing various configurations.
    :param datasetname: The name of the dataset to load.
    :param language: The language of the data to load.
    :return: The loaded dataset.
    """
    if datasetname == 'repoeval' and language != 'func_level':
        data_frame1 = pd.read_parquet(f"data/{datasetname}/{language}/test_0.parquet")
        data_frame2 = pd.read_parquet(f"data/{datasetname}/{language}/test_1.parquet")
        data_frame = pd.concat([data_frame1, data_frame2], ignore_index=True)
    else:
        data_frame = pd.read_parquet(f"data/{datasetname}/{language}/test.parquet")

    # data_frame = data_frame.loc[data_frame['task_id'] == 'project_cc_python/210']
    
    if datasetname == 'repoeval':
        language = 'python'

    if args.debug:
        data_frame = data_frame.sample(100)
    dataset = []
    for item in data_frame[["task_id", "path", "left_context", "right_context", "crossfile_context", "groundtruth"]].values:
        cross_files = item[4] if len(item[4]) > 0 else [{'path': "", "text": "Don't need cross file context for completion"}]
        cross_files = [CodeBlock(x["path"], f"file path: {x['path']}\nlines: {0}-{len(x['text'].splitlines())}", x["text"], language, '') for x in cross_files]
        dataset.append(Example(item[0], item[1], item[2], item[3], cross_files, item[5], language))
    
    return dataset

def load_train_and_valid_dataset():
    """
    Loads the training dataset.
    :return: The training dataset.
    """
    training_datasets = []
    validation_datasets = []
    for language in ["python", "java"]:
        data_frame = pd.read_parquet(f"data/github_repos/{language}/train.parquet")
        all_data = []
        temp_data = []
        for x in data_frame[["path", "content", "first"]].values:
            if x[-1]:  # At the start of a new file
                if len(temp_data) > 1:
                    all_data.append((temp_data,language))
                temp_data = []
            temp_data.append([x[0], x[1]])
        training_datasets.extend(all_data[:2000])
        validation_datasets.extend(all_data[2000:2200])
    random.shuffle(training_datasets)
    random.shuffle(validation_datasets)

    return training_datasets, validation_datasets


def construct_dataset(raw_data, num_samples):
    """
    Builds a dataset.
    :param raw_data: Raw data.
    :param num_samples: The number of samples to generate.
    :return: The list of constructed samples.
    """
    examples = []
    data_index = 0
    while len(examples) < num_samples:
        example,language = raw_data[data_index % len(raw_data)]
        data_index += 1
        selected_file = random.choice(example[1:])
        related_files = [CodeBlock(x[0], f"file path: {x[0]}\nlines: {0}-{len(x[1].splitlines())}", x[1], language, '') for x in example if x[0] != selected_file[0]]
        path = selected_file[0]
        selected_file_content = selected_file[1].split(" ")
        try_count = 0

        while try_count < 10:
            end_line_number = int(len(selected_file_content) * random.uniform(0.2, 0.8))
            left_context = " ".join(selected_file_content[:end_line_number])
            target_length = random.randint(32, 64)
            target = " ".join(selected_file_content[end_line_number:end_line_number + target_length])
            right_context = " ".join(selected_file_content[end_line_number + target_length:])
            if len(left_context.split()) > 80 and len(target.split()) > 8:
                examples.append(
                    Example(len(examples), path, left_context, right_context, related_files, target,language)
                )
                break
            # if language == 'python':
            #     if len(left_context.split()) > 64 and len(target.split()) > 5:
            #         examples.append(
            #             Example(len(examples), path, left_context, right_context, related_files, target,language)
            #         )
            #         break
            # elif language == 'java':
            #     if len(left_context.split()) > 80 and len(target.split()) > 8:
            #         examples.append(
            #             Example(len(examples), path, left_context, right_context, related_files, target,language)
            #         )
            #         break
            try_count += 1
    
    return examples