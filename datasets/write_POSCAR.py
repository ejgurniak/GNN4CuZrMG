import numpy as np
tol = (10)**(-5)
print(tol)

file_dump = open('dump.quench.cfg', 'r')
header_dump = []
list_dump = []
for i in range(17):
    myline = file_dump.readline()
    mylist = myline.split()
    header_dump.append(mylist)

num_atom = int(header_dump[0][4])
print("num_atom = " + str(num_atom))

for i in range(3*num_atom):
    myline = file_dump.readline()
    mylist = myline.split()
    list_dump.append(mylist)

file_dump.close()

Lx = -1.0
Ly = -2.0
Lz = -3.0
if header_dump[2][0] == 'H0(1,1)':
    Lx = float(header_dump[2][2])
else:
    print("ERROR: need to edit code because Lx is not found")
if header_dump[6][0] == 'H0(2,2)':
    Ly = float(header_dump[6][2])
else:
    print("ERROR: need to edit code because Ly is not found")
if header_dump[10][0] == 'H0(3,3)':
    Lz = float(header_dump[10][2])
if Lx < 0.0 or Ly < 0.0 or Lz < 0.0:
    print("ERROR: Lx, Ly, and/or Lz not initialized")

print("Lx = " + str(Lx))
print("Ly = " + str(Ly))
print("Lz = " + str(Lz))

# let's put Lx, Ly, Lz into an array for convenience later
box_length = np.array([[0.0]*3]*3)
box_length[0][0] = Lx
box_length[1][1] = Ly
box_length[2][2] = Lz
print(box_length)

data_Cu = np.array([[0.0]*4]*num_atom)
data_Zr = np.array([[0.0]*4]*num_atom)

num_Cu = 0
num_Zr = 0
for i in range(num_atom):
    mass_index = 3*i
    species_index = 3*i + 1
    data_index = 3*i + 2
    x = float(list_dump[data_index][0])
    y = float(list_dump[data_index][1])
    z = float(list_dump[data_index][2])
    if list_dump[species_index][0] == 'Cu':
        data_Cu[num_Cu][0] = x
        data_Cu[num_Cu][1] = y
        data_Cu[num_Cu][2] = z
        num_Cu += 1
    elif list_dump[species_index][0] == 'Zr':
        data_Zr[num_Zr][0] = x
        data_Zr[num_Zr][1] = y
        data_Zr[num_Zr][2] = z
        num_Zr += 1

# write to a new file
file_out = open('test1026.POSCAR', 'w')
file_out.write('label')
file_out.write("\n")
file_out.write('1.0')
file_out.write("\n")
for i in range(3):
    for j in range(3):
        file_out.write(str(box_length[i][j]))
        file_out.write(' ')
    file_out.write("\n")
file_out.write('Cu Zr')
file_out.write("\n")
file_out.write(str(num_Cu))
file_out.write(' ')
file_out.write(str(num_Zr))
file_out.write("\n")
file_out.write("direct")
file_out.write("\n")
for i in range(num_Cu):
    for j in range(3):
        file_out.write(str(data_Cu[i][j]))
        file_out.write(' ')
    file_out.write('Cu')
    file_out.write("\n")
for i in range(num_Zr):
    for j in range(3):
        file_out.write(str(data_Zr[i][j]))
        file_out.write(' ')
    file_out.write('Zr')
    file_out.write("\n")
file_out.close()
