def PISCOWanModelDictConverter(state_dict):
    state_dict_ = {name: state_dict[name] for name in state_dict if name.startswith("pisco")}
    return state_dict_
