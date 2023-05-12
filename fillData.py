import csv

for fileNumber in range(1, 10):
    writeFile = open(f'temperature_reef_by_month_1_9/{fileNumber}.0_filled.csv', 'w')
    filename = f'temperature_reef_by_month_1_9/{fileNumber}.0.csv'
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # Read the first row as headers
        for i in range(len(headers)):
            s = headers[i]
            if (',' in s):
                s = '\"' + s + '\"'
            writeFile.write(s)
            if (i + 1 != len(headers)):
                writeFile.write(',')
        writeFile.write('\n')
        for row in csvreader:
            columns = [row[i] for i in range(len(headers))]
            for i in range(42, 342):
                if (columns[i] != ''):
                    continue
                found = False
                startColumn = i
                endColumn = 2021 - int(headers[i][:4])
                for j in range(1, endColumn + 1):
                    if (columns[startColumn + (j * 12)] != ''):
                        columns[startColumn] = columns[startColumn + (j * 12)]
                        found = True
                        break
                if (found):
                    continue
                beforeStartColumn = int(headers[i][:4]) - 1997
                for j in range(1, beforeStartColumn + 1):
                    if (columns[startColumn - (j * 12)] != ''):
                        columns[startColumn] = columns[startColumn - (j * 12)]
                        found = True
                        break
            for i in range(len(columns)):
                s = columns[i]
                if (',' in s):
                    s = '\"' + s + '\"'
                writeFile.write(s)
                if (i + 1 != len(columns)):
                    writeFile.write(',')
            writeFile.write('\n')
    writeFile.close()

writeFile = open('data.csv', 'w')
writeFile.write('Site,Year,Depth,Coverage,Min Temp,Max Temp,Avg Temp,Med Temp,Prev Coverage\n')
for fileNumber in range(1, 6):
    filename = f'temperature_reef_by_month_1_9/{fileNumber}.0_filled.csv'
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # Read the first row as headers
        for row in csvreader:
            columns = [row[i] for i in range(len(headers))]
            for i in range(17, 42):
                column = i
                add1 = 2021 - int(headers[column]) + 1
                add2 = (12 * (int(headers[column]) - 1997))
                coverage = columns[column][:2]
                if (coverage == '' or len(coverage) != 2):
                    continue
                prevCoverage = None
                if (column == 17):
                    prevCoverage = coverage
                else:
                    prevCoverage = columns[column - 1][:2]
                    if (prevCoverage == '' or len(prevCoverage) != 2):
                        prevCoverage = coverage
                temps = []
                for i in range(0, 12):
                    temp = columns[column + add1 + add2 + i]
                    if (temp == ''):
                        continue
                    try:
                        temp = float(temp)
                    except:
                        print(column)
                        print(fileNumber)
                    temps.append(temp)
                if (len(temps) == 0):
                    continue
                avg = sum(temps) / len(temps)
                temps.sort()
                mid = len(temps) // 2
                median = (temps[mid] + temps[~mid]) / 2
                writeFile.write(f'\"{columns[0]}\",{headers[column]},{columns[4]},{coverage},{min(temps)},{max(temps)},{avg},{median},{prevCoverage}\n')

writeFile.close()
