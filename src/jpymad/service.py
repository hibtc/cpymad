'''
performs all the initialization stuff to access JMad from python.

Created on Nov 11, 2010

@author: kaifox
'''

# py4j for communicating with jmad
from py4j.java_gateway import JavaGateway

from variables import Enums

class PyMadService:
    """
    the service which is the main facade
    """
    def __init__(self):
        self.java_gateway = None
        self.jmad_service = None

    def is_connected(self):
        """
        returns true, if a connection is established, false if not.
        """
        return not (self.jmad_service == None)
    
    def _connect(self):
        """
        Creates the gateway and the jmad service
        """
        
        # if the java-gateway was already initialized, then we have nothing to do!
        if self.is_connected():
            return
        
        if self.java_gateway == None:
            self.java_gateway = JavaGateway()
        
        if self.jmad_service == None:
            # the entry-point is directly the jmad service
            # test the connection
            try:
                str = self.java_gateway.jvm.java.lang.String
            except:
                raise
            else:
                # only assign the jmad_service, if no error occured
                self.jmad_service = self.java_gateway.entry_point
        
        # now, that everything is connected, we can init the variables
        self.enums = Enums(self.java_gateway)

    
    def connect(self):
        """
        connects to an existing jmad-service
        """
        if self.is_connected():
            return
            
        self._connect()
    
# The singleton instance
pms = PyMadService()

def connect():
    """
    convenience method for pms.connect()
    """
    pms.connect()

