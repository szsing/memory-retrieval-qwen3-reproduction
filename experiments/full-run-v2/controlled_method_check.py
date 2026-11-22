"""Focused check that the fixed scoring implementation can use query-side properties."""
from night_methods import *
# Equal raw similarity; original query embedding matches donor 1, hidden query property matches target 0.
dense=np.array([.2,.2,.2]);lv=[np.array([[1.,0.]]),np.array([[0.,1.]]),np.array([[.6,.8]])]
qe=np.array([[0.,1.]]);qprops=np.array([[1.,0.]])
both=ranked(rrf(dense)+2*rrf(match(lv,qprops)))
write=ranked(rrf(dense)+2*rrf(match(lv,qe)))
prior=rrf(dense)+2*rrf(np.ones(3))
assert both[0]==0 and write[0]==1 and np.allclose(prior,prior[0])
# No extracted properties must confer equal, not position-dependent, RRF bonuses.
assert np.allclose(rrf(np.zeros(9)),rrf(np.zeros(9))[0])
print(json.dumps({'passed':True,'both_ranking':both,'write_ranking':write,'fact_prior_ties_preserved':True,'empty_properties_ties_preserved':True}))
