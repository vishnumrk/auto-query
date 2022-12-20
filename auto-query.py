import argparse
import csv
import itertools
import pathlib
import subprocess
import sys
import pandas as pd


#################################################################################################################
# python script_new.py -f test1.csv -t TableName -pt TableName -l 5 -c rec_seq_num -u -sd 2012-02-12 -ed 2021-02-03 #
#################################################################################################################
class ExpandFile:
    @staticmethod
    def expand(input_csv, cols=[]):
        return ExpandFile(input_csv, cols).expand_file()

    def __init__(self, input_csv, cols=[]):
        self.__hard_coded_cols = cols
        self.__input_csv = pathlib.Path(input_csv)
        self.__output_csv = self.__input_csv.with_name(
            self.__input_csv.stem + "_expanded.csv")
        reader = csv.DictReader(self.__input_csv.open())
        # hard coded columns has to be selected separately
        self.__column_names = reader.fieldnames
        self.__rows = [{k: v for k, v in row.items()} for row in reader]

    def expand_file(self):
        final_result = []
        for row in self.__rows:
            row_without_empty_value_cols = {k: v for k, v in row.items() if v}
            row_with_empty_value_cols = {k: v for k, v in row.items() if not v}
            cols_with_multiple_vals = [(col, [s.strip(None) for s in val.split(",")])
                                       for col, val in row_without_empty_value_cols.items()
                                       if self.__has_multiple_vals(val) and col]
            cols_with_single_vals = [(col, val) for col, val in row_without_empty_value_cols.items()
                                     if not self.__has_multiple_vals(val) and col]

            # [
            #   ('col_name1', [ 'val1', 'val2', 'val3']),
            #   ('col_name2', [ 'val1', 'val2', 'val3'])
            # ]
            # to
            # [
            #   [('col_name1','val1'), ('col_name1', 'val2'), ('col_name1', 'val3')],
            #   [('col_name2','val1'), ('col_name2', 'val2'), ('col_name2', 'val3')]
            # ]
            expand_multi_col_vals = [[(col_name, val) for val in value] for col_name, value in
                                     cols_with_multiple_vals]
            # [ [('col_name1','val1'), ('col_name2', 'val1')] ]
            expanded_records = [list(itertools.chain(list(item), cols_with_single_vals)) for item in
                                (itertools.product(*expand_multi_col_vals))]

            # [ { 'col_name1':'val1'}, {'col_name2': 'val1'} ]
            expanded_records = [{k: v for k, v in rec}
                                for rec in expanded_records]
            expanded_records = self.index_hard_coded_values(expanded_records)
            expanded_records = [{**rec, **row_with_empty_value_cols}
                                for rec in expanded_records]
            final_result.extend(expanded_records)

        writer = csv.DictWriter(self.__output_csv.open(
            'w', newline=''), self.__column_names)
        writer.writeheader()
        writer.writerows(final_result)
        return self.__output_csv

    def index_hard_coded_values(self, expanded_records):
        processed_matching_cols = {}
        for rec in expanded_records:
            for col_name in self.__hard_coded_cols:
                value = rec[col_name]
                if value in processed_matching_cols:
                    processed_matching_cols[value] = processed_matching_cols[value] + 1
                    value = f"{value}-{processed_matching_cols[value]}"
                else:
                    processed_matching_cols[value] = 0
                rec[col_name] = value
        return expanded_records

    @staticmethod
    def __has_multiple_vals(val):
        return len(str(val).split(',')) > 1


class AutoQuery:
    def __init__(self, args):
        input_file = pathlib.Path(args.file)
        if not input_file.exists():
            print(args.file, "does not exist. please provide a valid input file")
            sys.exit()
        if args.expand_query:
            input_file = ExpandFile.expand(
                args.file, args.cols if args.cols else [])

        self.__input_csv = args.file
        self.__table = args.table
        self.__pre_map_table = args.pre_map_table
        self.__hard_coded_cols = args.cols if args.cols else []
        self.__limit = args.limit
        self.__union = args.union
        self.__run_date_start = args.start_date
        self.__run_date_end = args.end_date
        self.__expand_query = args.expand_query
        self.__expanded_conditions = []
        reader = csv.DictReader(input_file.open())
        # hard coded columns has to be selected separately
        self.__cols_to_select = [
            col for col in reader.fieldnames if col not in self.__hard_coded_cols]
        self.__rows_dict_list = [
            {k: v for k, v in row.items() if v} for row in reader]
        self.__rows_without_records = []
        self.__match_col = self.__hard_coded_cols[0]

    def prepare_query(self):
        queries = self.__prepare()
        queries = [f"{query} and run_date between '{self.__run_date_start}' and '{self.__run_date_end}'"
                   for query in queries]
        queries = [
            f"{query} limit {self.__limit}" if self.__limit else query for query in queries]

        q_path = self.__write_to_input_path_with_suffix(
            '\n'.join(queries), "_queries.sql")
        queries_to_union = [
            f"select * from ({q}) q{i}" for i, q in enumerate(queries)]
        union_query = "\nunion all\n".join(queries_to_union)

        if self.__union:
            self.__write_to_input_path_with_suffix(
                union_query, "_union_queries.sql")

        query_out = q_path.absolute().with_name(q_path.stem + "_output.csv")
        # subprocess.call("echo " + union_query.replace("\n", " "), shell=True)
        sql_query = pd.read_sql_query(
            union_query, con='postgresql://postgres:password@localhost:5432/local')
        sql_query.to_csv(query_out, index=False)
        self.__post_process(query_out)
        self.__execute_pre_map_query()

    def __prepare(self):
        queries = []
        for row in self.__rows_dict_list:
            select_cols = [
                f"'{row.get(col)}' as {col}" for col in self.__hard_coded_cols]
            select_cols = list(itertools.chain(
                select_cols, self.__cols_to_select))

            # {'col_name1': 'val1,val2,val3', 'col_name2': 'val1,val2,val3'}
            # to
            # [
            #   ('col_name1', [ 'val1', 'val2', 'val3']),
            #   ('col_name2', [ 'val1', 'val2', 'val3'])
            # ]
            cols_with_multiple_vals = [(col, [s.strip(None) for s in val.split(",")])
                                       for col, val in row.items()
                                       if self.__has_multiple_vals(val) and col not in self.__hard_coded_cols]
            cols_with_single_vals = [(col, val) for col, val in row.items() if
                                     not self.__has_multiple_vals(val) and col not in self.__hard_coded_cols]

            conditions_for_single_val_cols = [
                f"{key}='{val}'" for (key, val) in cols_with_single_vals]
            conditions_for_multi_val_cols = [f"""{key} in ({",".join([f"'{v}'" for v in val])})"""
                                             for (key, val) in cols_with_multiple_vals]
            conditions_for_multi_val_cols.extend(
                conditions_for_single_val_cols)
            queries.append(self.__form_query(
                select_cols, self.__table, conditions_for_multi_val_cols))
        return queries

    def __execute_pre_map_query(self):
        queries = [
            f"select '{row[self.__match_col]}' as school_code_m,* from {self.__pre_map_table} where run_date='{row['run_date']}' and billing_line='{row['billing_line']}' and school_code='{row['school_code']}'"
            for row in self.__pre_map_input
        ]

        if not queries:
            print("Pre-map queries are empty")
            return
        sql_query = pd.read_sql_query(" union all ".join(queries),
                                      con='postgresql://postgres:password@localhost:5432/local')
        input_file_path = pathlib.Path(self.__input_csv)
        p = input_file_path.with_name(input_file_path.stem + "_final.csv")
        sql_query.to_csv(p, index=False)
        self.__post_process(p)

    def __post_process(self, query_out):
        match_col_values = [row[self.__match_col]
                            for row in self.__rows_dict_list]
        reader = csv.DictReader(query_out.open())
        vals = [{k: v for k, v in row.items()} for row in reader]
        out_csv_dict = {k: list(v) for k, v in itertools.groupby(
            vals, key=lambda x: x[self.__match_col])}
        self.__pre_map_input = [
            item for sublist in out_csv_dict.values() for item in sublist]
        out_file_grouped = query_out.with_name(query_out.stem + "_grouped.csv")
        for val in match_col_values:
            csvfile = csv.writer(out_file_grouped.open("a", newline=''))
            csvfile.writerow([val])
            if val in out_csv_dict:
                flag = True
                for r in out_csv_dict[val]:
                    if flag:
                        csvfile.writerow(r.keys())
                        flag = False
                    csvfile.writerow(r.values())
            else:
                self.__rows_without_records.append(val)
                csvfile.writerow(["No records found"])

    def __write_to_input_path_with_suffix(self, data, suffix):
        input_file_path = pathlib.Path(self.__input_csv)
        p = input_file_path.with_name(input_file_path.stem + suffix)
        p.write_bytes(str.encode(data))
        return p

    @staticmethod
    def __form_query(cols, table, conditions):
        select_cols = ",".join(cols)
        condition = " and ".join(conditions)
        return f"select {select_cols} from {table} where {condition}"

    @staticmethod
    def __has_multiple_vals(val):
        return len(str(val).split(',')) > 1

    @staticmethod
    def run():
        # Initialize parser
        parser = argparse.ArgumentParser()

        # Adding mandatory arguments
        parser.add_argument(
            "-f", "--file", help="input csv file", required=True)
        parser.add_argument(
            "-t", "--table", help="table name to prepare the query", required=True)
        parser.add_argument("-pt", "--pre-map-table",
                            help="table name to do pre map query", required=True)
        parser.add_argument(
            "-sd", "--start-date", help="start date for run_date in 'yyyy-mm-dd' format", required=True)
        parser.add_argument(
            "-ed", "--end-date", help="end date for run_date in 'yyyy-mm-dd' format", required=True)
        # Adding optional arguments
        parser.add_argument("-l", "--limit", type=int,
                            help="add limit to queries", default=10)
        parser.add_argument(
            "-c", "--cols", help="columns for which the values are hard coded in the file", nargs='+')
        parser.add_argument("-u", "--union", action='store_true',
                            help="get union of all queries")
        parser.add_argument("-eq", "--expand-query", action='store_true', default=False,
                            help="expand the query to multiple queries if any column has ',' seperated values. If "
                                 "provided 'false' then query will be created with an in clause")

        # Read arguments from command line
        arguments = parser.parse_args()
        auto_query = AutoQuery(arguments)
        auto_query.prepare_query()


if __name__ == '__main__':
    AutoQuery.run()
