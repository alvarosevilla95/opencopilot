from IPython.core.getipython import get_ipython

def create_new_cell(contents):
    payload = dict(source='set_next_input', text=contents, replace=False)
    get_ipython().payload_manager.write_payload(payload, single=False)

