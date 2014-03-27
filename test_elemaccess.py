from cern import cpymad
lhc=cpymad.model('lhc')
print lhc.get_sequences()
all_elements=lhc.get_element_list('lhcb1')
print lhc.get_element('lhcb1',all_elements[3])

