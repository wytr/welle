proximityValue = 10
array = [225,226,227,291,293]
found_clusters = []

for value in array:
    cluster = [value]
    array.remove(value)
    for pos in array:
        print(f"{pos} - {value} = {pos - value}")
        if pos-value<10:
            cluster.append(pos)
    
    for element in cluster[1:]:
        array.remove(element)
    found_clusters.append(cluster)
print(found_clusters)
        