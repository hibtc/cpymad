
from cern.cpymad._couch import couch
import os,json

overwrite=True

def uploadModel(c,mod,modname,**kwargs):
    d=json.loads(file(mod,'r').read())
    print("Uploading "+modname)
    c.put_model(modname,d,**kwargs)



def uploadJmadModels():
    c=couch.Server(dbname='jmad_models')

    avail_models=c.ls_models()
    #for mod in avail_models:
        #c.del_model(mod)
    for mod in os.listdir('.'):
        if mod[-9:]=='.jmd.json':
            if overwrite or mod not in avail_models :
                uploadModel(c,mod,mod[:-9])


def _convertBeamOrTWinit(ds,s):
        for val in s:
            if val=='@name':
                ds['name']=s[val]
            else:
                v=s[val]['@value']
                if v=='false':
                    ds[val]=False
                elif v=='true':
                    ds[val]=True
                else:
                    try:
                        ds[val]=float(v)
                    except ValueError: # not a number..
                        ds[val]=v

def jmad2cpymad(mod,usecouch=True):
    '''
     converts jmad model def from 
     couch db to cpymad model def
    '''
    if usecouch:
        c=couch.Server(dbname='jmad_models')
        d=c.get_model(mod)
    else:
        d=json.loads(file(mod+'.jmd.json','r').read())
    dnew={}
    
    # some defaults:
    dnew["default"]={}
    dnew["dbdirs"]=['/afs/cern.ch/eng/']
    
    dnew['name']=d['jmad-model-definition']['@name']
    
    jd=d['jmad-model-definition']
    
    dnew["default"]['optic']=jd['default-optic']['@ref-name']
    
    # init files
    inits=jd['init-files']['call-file']
    dnew["init-files"]=[]
    for f in inits:
        fnew={"path":f["@path"]}
        if "@location" in f:
            fnew["location"] = f["@location"].lower()
        else:
            fnew["location"] = "repository"
        dnew["init-files"].append(fnew)
    
    # folder paths
    jpo=jd['path-offsets']
    dnew["path-offsets"]={"resource":jpo['resource-offset']['@value']}
    if "repository-offset" in jpo:
        repoff=jpo['repository-offset']['@value']
    else:
        repoff=''
    dnew['path-offsets']['repository']=repoff
    
    # sequences
    dnew['sequences']={}
    ds=dnew['sequences']
    for s in jd['sequences']['sequence']:
        # we just select the first sequence as default..
        if not 'sequence' in dnew['default']:
            dnew['default']['sequence']=s['@name']
        #print s.keys()
        dnew['sequences'][s['@name']]={}
        ds=dnew['sequences'][s['@name']]
        
        # not defined in jmad:
        ds['aperfiles']=[]
        ds['offsets']=[]
        
        # ranges:
        ds['default-range']=s['default-range']['@ref-name']
        ds['ranges']={}
        dsr=ds['ranges']
        for r in s['ranges']['range']:
            dsr[r['@name']]={}
            dsr[r['@name']]['range']=[r['madx-range']['@first'],r['madx-range']['@last']]
            dsr[r['@name']]['twiss-init']={}
            _convertBeamOrTWinit(dsr[r['@name']]['twiss-init'],r['twiss-initial-conditions'])
        # beam:
        ds['beam']={}
        _convertBeamOrTWinit(ds['beam'],s['beam'])
    
    # optics
    dnew['optics']={}
    for o in jd['optics']['optic']:
        dnew['optics'][o['@name']]={}
        do=dnew['optics'][o['@name']]
        if o['@overlay']=='true':
            do['overlay']=True
        else:
            do['overlay']=False
        do['strengths']=[]
        for s in o['init-files']['call-file']:
            onlyOne=False
            if type(s)!=dict:
                s=o['init-files']['call-file']
                onlyOne=True
            f={}
            f['path']= s['@path']
            if '@location' in s:
                f['location']=s['@location'].lower()
            else:
                f['location']='repository'
            
            do['strengths'].append(f)
            if onlyOne:
                break
    
    file(mod+'.cpymad.json','w').write(json.dumps(dnew,indent=2))
    if usecouch:
        fnames,fpaths=_get_file_list(dnew)
        c2=couch.Server(dbname='cpymad_newstyle')
        uploadModel(c2,mod+'.cpymad.json',mod,fnames=fnames,fpaths=fpaths)

def convertModels(usecouch=True):
    if usecouch:
        c=couch.Server(dbname='jmad_models')
        avail_models=c.ls_models()
    else:
        avail_models=[]
        for f in os.listdir('.'):
            if f[-9:]=='.jmd.json':
                avail_models.append(f[:-9])
    for mod in avail_models:
        print "Converting "+mod+'..'
        #try:
        jmad2cpymad(mod, usecouch)
        #except TypeError:
            #print " FAILED"

def _append_file(f,fnames,fpaths,offsets,locations):
    fnames.append(offsets[f['location']]+'/'+f['path'])
    fpaths.append(locations[f['location']]+'/'+fnames[-1])
    #if not os.path.isfile(fpaths[-1]):
        #raise ValueError("Missing file %s" % fpaths[-1])

def _get_file_list(mdict):
    from cern import cpymad
    
    dbdir=''
    fpaths=[]
    fnames=[]
    for d in mdict['dbdirs']:
        if os.path.isdir(d):
            dbdir=d
            break
    if not dbdir:
        raise ValueError("Could not find a valid db directory")
    locations={"repository":dbdir,"resource": os.path.dirname(cpymad.__file__)+'/_models/resdata/'}
    print mdict['path-offsets']
    for f in mdict['init-files']:
        _append_file(f,fnames,fpaths,mdict['path-offsets'],locations)
    
    for seq in mdict['sequences']:
        for f in mdict['sequences'][seq]['aperfiles']:
            _append_file(f,fnames,fpaths,mdict['path-offsets'],locations)
        for f in mdict['sequences'][seq]['offsets']:
            _append_file(f,fnames,fpaths,mdict['path-offsets'],locations)
            
    for op in mdict['optics']:
        for f in mdict['optics'][op]['strengths']:
            _append_file(f,fnames,fpaths,mdict['path-offsets'],locations)
    return fnames,fpaths
        
if __name__=="__main__":
    uploadJmadModels()
    print "Converting lhc"
    jmad2cpymad('lhc',usecouch=True)
    #convertModels(usecouch=True)
    #_get_file_list(json.loads(file('lhc.cpymad.json','r').read()))
