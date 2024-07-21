import ast
import re
import pandas as pd
from prettytable import PrettyTable


class FunctionDefExtractor(ast.NodeVisitor):
    def __init__(self, source_code):
        self.source_code = source_code.split('\n')
        self.functions = []

    def visit_FunctionDef(self, node):
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', start_line + 1)
        if end_line - start_line > 20:
            end_line = start_line + 20
        function_code = '\n'.join(self.source_code[start_line:end_line])
        self.functions.append((start_line, end_line, function_code))
        self.generic_visit(node)

def extract_python_functions(code_content):
    try:
        tree = ast.parse(code_content)
        extractor = FunctionDefExtractor(code_content)
        extractor.visit(tree)
        return extractor.functions
    except:
        []

def extract_java_methods(source_code):
    method_pattern = re.compile(r'(public|protected|private|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(\{?|[^;])', re.DOTALL)
    methods = []
    lines = source_code.split('\n')
    for match in method_pattern.finditer(source_code):
        start_line = source_code[:match.start()].count('\n')
        method_content = match.group()
        method_lines = method_content.count('\n') + 1
        end_line = start_line + method_lines
        if method_lines > 20:
            method_content = '\n'.join(method_content.split('\n')[:20])
            end_line = start_line + 20
        methods.append((start_line, end_line, method_content))
    return methods

def sort_prettytable(table, by=['Method']):
    rows = table.get_rows()
    data = [list(row) for row in rows]
    columns = table.field_names

    df = pd.DataFrame(data, columns=columns)
    df_sorted = df.sort_values(by=by)

    sorted_table = PrettyTable()
    sorted_table.field_names = columns
    for index, row in df_sorted.iterrows():
        sorted_table.add_row(row.values)
    
    return sorted_table



def extract_import_statements(examples, context_len):
    extracted_imports = []
    
    for example in examples:

        lines = example.left_context.split('\n')
        import_lines = []
        

        if example.language == 'python':

            import_lines = [line for line in lines if line.strip().startswith('import ') or line.strip().startswith('from ') and 'import' in line]
        elif example.language == 'java':

            import_lines = [line for line in lines if line.strip().startswith('import ')]
        
        import_lines = '\n'.join(import_lines[-context_len:])
        extracted_imports.append(import_lines)
    
    return extracted_imports