
import json,os

# some fixed conversions:
_VALUE_MAP = {'true' : True, 'false' : False, 'PLUS': 1, 'MINUS': -1}

# in this case we want to "push down" the name
_NAME_CONVERT_LIST=['jmad-model-definition','sequence']
# in this case we want to "push down" the name, but keep
# the parent as a dicitonary as well..
_NAME_CONVERT_LIST_KEEP_PARENT=['twiss-initial-conditions']

def _convert_key(key):
    if key.startswith('@'):
        return key[1:]
    else:
        return key

def _try_float_convert(value):
    try:
        return float(value)
    except:
        return None

def _convert_value(value):
    if _VALUE_MAP.has_key(value):
        return _VALUE_MAP[value]
    
    newval = _try_float_convert(value)
    if not newval == None:
        return newval
    
    return value

def _convert_dict_keep_parent(newdict,value,thiskey):
    newdict[thiskey]={}
    thiskey2=value['@name']
    del value['@name']
    newdict[thiskey][thiskey2] = _convert_recursively(value)
    return newdict

def _convert_recursively(item):
    '''
    Recursively convert the model from jmad...
    '''
    if  isinstance(item, list):
        convert2dict=True
        for nextItem in item:
            if not '@name' in nextItem:
                convert2dict=False
        if convert2dict:
            newdict = {}
            for nextItem in item:
                key=nextItem['@name']
                del nextItem['@name']
                newdict[key]=_convert_recursively(nextItem)
            return newdict
        else:
            newlist = []
            for nextItem in item:
                newlist.append(_convert_recursively(nextItem));
            return newlist
    elif isinstance(item, dict):
        newdict = {}    
        for key, value in item.items():
            thiskey=_convert_key(key)
            if '@name' in value and thiskey in _NAME_CONVERT_LIST:
                thiskey=value['@name']
                del value['@name']
            elif '@name' in value and thiskey in _NAME_CONVERT_LIST_KEEP_PARENT:
                return _convert_dict_keep_parent(newdict,value,thiskey)
            elif thiskey=='init-files' and 'call-file' in value:
                value=value['call-file']
            elif '@value' in value and len(value)==1:
                value=value['@value']
            elif thiskey.split('-')[0]=='default':
                if '@ref-name' in value:
                    value=value['@ref-name']
            newdict[thiskey] = _convert_recursively(value)
        return newdict
    else:
        return _convert_value(item)

def _add_default_cpymads(new_dict):
    '''
     Add some default stuff needed in the cpymad
     definition.
    '''
    for mname,model in new_dict.items():
        model['dbdirs']=['/afs/cern.ch/eng/']
        for seqname,sequence in model['sequences'].items():
            sequence['aperfiles']=[]

def _move_beams(new_dict):
    '''
     Moving beams in the dictionary..
    '''
    for mname,model in new_dict.items():
        model['beams']={}
        for seqname,sequence in model['sequences'].items():
            if 'beam' not in sequence:
                print("WARNING: No beam defined for "+seqname)
                continue
            if mname+'_'+seqname in model['beams']:
                raise ValueError("Two beams with same name, please resolve")
            model['beams'][mname+'_'+seqname]=sequence['beam']
            model['beams'][mname+'_'+seqname]['sequence']=seqname
            sequence['beam']=mname+'_'+seqname
            for r in sequence['ranges']:
                if 'default-twiss' not in sequence['ranges'][r]:
                    sequence['ranges'][r]['default-twiss']=sequence['ranges'][r]['twiss-initial-conditions'].keys()[0]

def convert_dict(indict):
    '''
    converts a jmad-model definition to one which is more nicely readble from cpymad
    '''
    new_dict=_convert_recursively(indict)
    _add_default_cpymads(new_dict)
    _move_beams(new_dict)
    return new_dict

def convert_file(infilename, outfilename):
    indict = json.loads(file(infilename, 'r').read())
    outdict = convert_dict(indict);
    file(outfilename, 'w').write(json.dumps(outdict, indent=2))

if __name__ == "__main__":
    skip=['lhc','longti8']
    for f in os.listdir('.'):
        if f[-9:]=='.jmd.json' and f[:-9] not in skip:
            print("Converting "+f[:-9])
            convert_file(f, '../_models/'+f[:-9]+'.cpymad.json')
            
            # saving jmad file in pretty print format:
            #jd=json.load(file(f,'r'))
            #json.dump(jd,file(f,'w'),indent=2)
    
