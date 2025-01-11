dict_items={'apple':2,'banana':3,'cherry':1}
print(dict_items.items())

sorted_dict = sorted(dict_items.items(),key=lambda item:item[1])
print(sorted_dict)

numbers = [1,2,3,4,5,6]
evens = list(filter(lambda x:x %2 ==0,numbers))
print(evens)


numbers = [1,2,3,4,5,6]

squared = list(map(lambda x : x**2,numbers))
print(squared)
