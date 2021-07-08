int input_file_reader(){

  FILE *fp;
  char *filename = "input.csv";
  char readline[256] = {'\0'};

  if ((fp = fopen(filename, "r")) == NULL) {
    fprintf(stderr, "file open failed (%s).\n", filename);
    exit(EXIT_FAILURE);
  }

  while ( fgets(readline, 256, fp) != NULL ) {
    char *tp;
    tp = strtok( readline, " ." );
    puts( tp );
    while ( tp != NULL ) {
      tp = strtok( NULL," ." );
      if ( tp != NULL ) puts( tp );
    }
    printf("\n");

    return 0;
  }

  fclose(fp);
}
