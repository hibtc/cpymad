
import json,yaml,os


def _decode_list(data):
    '''
    Borrowed from
    http://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-ones-from-json-in-python

    Need this to get normal string objects in the dictionary instead of unicode ones
    (which makes the yaml file much harder to read)
    '''
    rv = []
    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv

def _decode_dict(data):
    '''
    Borrowed from
    http://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-ones-from-json-in-python

    Need this to get normal string objects in the dictionary instead of unicode ones
    (which makes the yaml file much harder to read)
    '''
    rv = {}
    for key, value in data.iteritems():
        if isinstance(key, unicode):
            key = key.encode('utf-8')
        if isinstance(value, unicode):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = _decode_list(value)
        elif isinstance(value, dict):
            value = _decode_dict(value)
        rv[key] = value
    return rv


if __name__ == "__main__":
    folder='../_models/'
    for f in os.listdir(folder):
        if f[-12:]=='.cpymad.json':
            print("Converting "+f[:-12])
            d=json.load(file(folder+f),object_hook=_decode_dict)
            yaml.dump(d,stream=file(folder+f[:-4]+'yml','w'))
