##Introducir dos numeros por teclado. Imprimir los numeros que hay entre ellos 
###comenzando por el mas pequeno. Contar cuantos hay y cuantos de ellos son 
###pares. Calcular la suma de los pares
p = 0
cp = 0
c = 0
n = 0
h = 0
h1 = input('Primer numero: ')
h2 = input('Segundo numero: ') 
if h1 > h2:
    n = h2
    h = h1
else:
    n = h1
    h = h2
while n < h:
    n += 1
    if n == h:
        break
    c += 1
    print n,
    if n%2 == 0:
        cp += 1
        p += n
print '\nEntre % i y %i hay %i numeros siendo %i pares' % (h1, h2, c, cp)
print 'La suma de los pares es %i' % p