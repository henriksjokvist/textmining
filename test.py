def number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

a = []

filepath = 'C:\Fel\Robinstest\Test.csv'

with open(filepath, 'r') as f:
    kommentar = f.readline()
    while(kommentar!=""):
        a.append(kommentar.strip('\n'))
        kommentar = f.readline()
f.closed

b = []

filepath = "C:\Fel\Robinstest\Forkortningar.csv"

with open(filepath, 'r') as fb:
    forkortning = fb.readline()
    while(forkortning!=""):
        b.append(forkortning)
        forkortning = fb.readline()
fb.closed

c = []
d = []
for k in b:
    c.append(k[0:k.find(";")].lower())
    d.append(k[k.find(";"):].strip("\n;"))
    
ImpC = []

for k in a:
    kom = ""
    i=0
    for t in k.replace('/',' ').split(" "):
        t = t.strip(',- ')
        print(t)
        if t.lower() in c:
            t = d[c.index(t.lower())]
            
            
        if t !="" and number(t) == False:
            if(i == 0):
                kom= t.replace('+','')
            else:
                kom = kom + " "+ t.replace('+','')
        i=i+1
    
    ImpC.append(kom)



filepath = 'C:/Fel/Robinstest/res.csv'

with open(filepath, 'w') as fc:
    fc.writelines("%s\n" % l for l in ImpC)
fc.closed







