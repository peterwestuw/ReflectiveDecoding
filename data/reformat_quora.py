with open('quora/quora_duplicate_questions.tsv') as f_in, open('quora/quora_questions.tsv','w') as f_out:
    for i, line in enumerate(f_in):
        
        ## skip the first line (banner)
        if i == 0:
            continue
            
            
        try:
            source, target = line.split('\t')[3],line.split('\t')[4]
        except:
            print('FAILED)
            print(line)
        f_out.write( '{}\t{}\n'.format(source,target))