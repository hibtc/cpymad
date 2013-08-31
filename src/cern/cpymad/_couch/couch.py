#-------------------------------------------------------------------------------
# This file is part of PyMad.
#
# Copyright (c) 2011, CERN. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#-------------------------------------------------------------------------------
import couchdb

class Server():
    def __init__(self,url='137.138.26.237',port=5984,user='jmad_user',password='iAmJmad99',dbname='cpymad_models'):
        self._couch = couchdb.Server('http://'+user+':'+password+'@'+url+':'+str(port))
        self._db = self._couch[dbname]

    def ls_models(self):
        ret=[]
        for doc in self._db:
            ret.append(str(doc))
        return ret

    def get_file(self,model,fname):
        '''
         Returns the content of a file
         in the form of a string.
        '''
        return self._db.get_attachment(model,fname)


    def ls_files(self,model):
        '''
         Returns a list of all files defined for a model.
        '''
        d=self._db[model]
        ret=[d['initscript']]
        for optic in d['optics']:
            ret.extend(optic['strengths'])
        return ret

    def get_model(self,model):
        '''
         Returns a model definition, which is a dictionary,
         or more precisely a couchdb document.
        '''
        return self._db[model]

    def put_model(self,modname,dictionary,**kwargs):
        '''
         Create a new model..

         kwargs:
         fnames: name of files as they should be on the server
         fpaths: corresponding full paths for each file on your machine
        '''
        check_model_valid(dictionary)

        if modname in self.ls_models():
            doc=self._db[modname]
            for k in dictionary:
                doc[k]=dictionary[k]
            dictionary=doc
        self._db[modname]=dictionary

        if 'fnames' in kwargs:
            if len(kwargs['fnames'])!=len(kwargs['fpaths']):
                raise ValueError("You need to give one filename for each attachment")
            for (a,f) in zip(kwargs['fnames'],kwargs['fpaths']):
                print "Uploading attachment %s" % a
                content=open(f,'r')
                self._db.put_attachment(self._db[modname], content, filename=a)

    def del_model(self,modname):
        self._db.delete(self._db[modname])

def check_model_valid(dictionary):
    '''
     We don't currently check..
    '''
    return True


