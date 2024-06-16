from dagster import ConfigurableIOManager, FilesystemIOManager, IOManager, io_manager, OutputContext, InputContext, OpExecutionContext
import os
import pickle

class CustomFilesystemIOManager(ConfigurableIOManager):
    
    base_dir: str = ""

    def _get_path(self, context):
        # graph_name = context.op_def.graph_name
        # op_name = context.op_def.name
        return f"{os.path.abspath(self.base_dir)}" #/{op_name}

    def handle_output(self, context: OutputContext, obj):
        asset_name = context.name
        path = f'{self._get_path(context)}/{asset_name}'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(obj, file)

    def load_input(self, context: InputContext):
        asset_name = context.name
        path = f'{self._get_path(context)}/{asset_name}'
        with open(path, 'rb') as file:
            return pickle.load(file)