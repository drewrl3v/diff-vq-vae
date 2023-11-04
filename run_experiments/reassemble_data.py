def split_file(file_name, chunk_size_mb=1024):
    ''' 
    This function was used to breakup the original dataset into chunks of 500MB
    '''
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert to bytes
    part_number = 1
    with open(file_name, 'rb') as f:
        chunk = f.read(chunk_size)
        while chunk:
            with open(f"{file_name}_part_{part_number}", 'wb') as chunk_file:
                chunk_file.write(chunk)
            part_number += 1
            chunk = f.read(chunk_size)

def join_files(original_file_name, number_of_parts):
    ''' 
    Join all the files chunks to form the original datasets
    '''
    with open(original_file_name, 'wb') as outfile:
        for part_number in range(1, number_of_parts + 1):
            with open(f"{original_file_name}_part_{part_number}", 'rb') as infile:
                outfile.write(infile.read())

join_files('./data_ml/processed_tracts.pt', number_of_parts=13)
join_files('./data_ml/val_set.pt', number_of_parts=13)