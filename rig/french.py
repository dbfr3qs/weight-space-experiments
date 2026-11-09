import re
FR=set("le la les un une des et est en du de que qui pour dans sur pas plus avec ce cette sont vous nous il elle ils au aux ou mais par se sa son ses leur très être avoir fait comme aussi peut tout bien".split())
EN=set("the a an and is are of to in that for on with this it you your be as by or but not can from at which more have has was will".split())
def score(text):
    w=re.findall(r"[a-zàâçéèêëîïôûùüÿœæ']+",text.lower())
    f=sum(x in FR for x in w); e=sum(x in EN for x in w)
    return f/(f+e) if f+e else 0.5
def is_french(text): return score(text)>0.6
