/* SECTION 1: DEFINITIONS AND SETUP */
%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define YYDEBUG 0
/* Forward declarations for the lexer and error functions. */
int yylex();
void yyerror(const char *s);

/* The lexer will provide the line number for error reporting. */
extern int yylineno;
extern FILE *yyin; /* Make the input file pointer available to main */

%}

/* The %union defines all possible data types that a token or grammar rule can hold. */
%union {
    char *stringValue;
    int intValue;
    float floatValue;
    /* We will add pointers to AST nodes here in the next stage. */
}

/* Assign types from the %union to all tokens that carry a value. */
%token <stringValue> T_IDENTIFIER T_BUILT_IN T_STRING_LITERAL T_CHAR_LITERAL
%token <intValue> T_INTEGER_LITERAL
%token <floatValue> T_FLOAT_LITERAL

/* Declare all other tokens that don't carry a specific value. */
%token T_TRUE T_FALSE T_UNKNOWN
%token T_GREATER_EQUAL T_LESS_EQUAL T_EQUAL_EQUAL T_NOT_EQUAL T_AND T_OR
%token T_METRIC_CLOSURE T_TRANSITIVE_CLOSURE T_TRANSPOSE T_COMPLEMENT T_CONNECTED_COMP T_POWER T_AMPERSAND T_TENSOR_PRODUCT
%token T_INT T_FLOAT T_CHAR T_STRING T_DOUBLE T_BOOL T_CONST T_VAR
%token T_VERTEX_SET T_EDGE_SET T_GRAPH T_FUNK T_DOT
%token T_IF T_ELIF T_ELSE T_FOR T_WHILE T_BREAK T_CONTINUE T_VECTOR T_STRUCT T_NEW T_RETURN T_IN T_RANGE T_WHERE

/* --- OPERATOR PRECEDENCE AND ASSOCIATIVITY --- */
/* This is the heart of the grammar. Lowest precedence is listed first. */
%right '='
%left T_OR
%left T_AND
%left T_EQUAL_EQUAL T_NOT_EQUAL
%left '<' '>' T_LESS_EQUAL T_GREATER_EQUAL
%left '+' '-'
%left '*' '/' '%'
%left T_AMPERSAND T_TENSOR_PRODUCT  /* Custom GraphX Binary Ops */
%right T_POWER
%right UNARY /* A virtual token for unary operators */
%left T_TRANSITIVE_CLOSURE T_METRIC_CLOSURE T_TRANSPOSE /* Postfix graph ops */
%left '.' '(' ')' '[' ']' /* Highest precedence for calls and access */


%%
/* SECTION 2: GRAMMAR RULES */

/* The root of our grammar: a program is one or more top-level declarations. */
program:   
			';'
    | /* Epsilon */
    | external_declaration_list
      ;

external_declaration_list:
      external_declaration
    | external_declaration_list external_declaration
    ;

external_declaration:
      declaration_statement
    | function_definition
    | struct_definition
    ;

/* A statement is a single "sentence" in the language. */
statement:
      declaration_statement
    | selection_statement
    | iteration_statement
    | expression_statement
    | compound_statement
    | jump_statement
    | ';' /* Empty statement */
    ;

statement_list:
      statement
    | statement_list statement
    ;

/* --- Declaration Statements --- */
declaration_statement:
      T_VAR T_IDENTIFIER optional_reference ':' type_specifier optional_initializer ';'
    | T_CONST T_IDENTIFIER optional_reference ':' type_specifier '=' expression ';'
    ;

optional_reference:
      T_AMPERSAND
    | /* epsilon */
    ;

type_specifier:
      T_INT | T_FLOAT | T_CHAR | T_STRING | T_DOUBLE | T_BOOL | T_VECTOR
    | T_STRUCT | T_VERTEX_SET | T_EDGE_SET | T_GRAPH | T_IDENTIFIER /* For user-defined struct types */
    ;

optional_initializer:
      '=' expression
    | /* epsilon */
    ;

/* --- Struct Definition --- */
struct_definition:
    T_STRUCT T_IDENTIFIER '{' struct_member_list '}' ';'
    ;

struct_member_list:
    declaration_statement
    | struct_member_list declaration_statement
    ;

/* --- Function Definition --- */
function_definition:
      T_FUNK T_IDENTIFIER '(' optional_parameter_list ')' ':' type_specifier compound_statement
      ;

optional_parameter_list:
      parameter_list
    | /* epsilon */
    ;

parameter_list:
      parameter_declaration
    | parameter_list ',' parameter_declaration
    ;

parameter_declaration:
      T_IDENTIFIER ':' type_specifier
      ;

/* A block of code enclosed in curly braces */
compound_statement:
      '{' optional_statement_list '}'
      ;

optional_statement_list:
      statement_list
    | /* epsilon */
    ;

/* --- Control Flow Statements --- */
selection_statement:
      T_IF '(' expression ')' statement
    | T_IF '(' expression ')' statement T_ELSE statement
    | T_IF '(' expression ')' statement T_ELIF '(' expression ')' statement
    | T_IF '(' expression ')' statement T_ELIF '(' expression ')' statement T_ELSE statement
    ;

iteration_statement:
      T_WHILE '(' expression ')' statement
    | T_FOR T_IDENTIFIER T_IN T_RANGE '(' expression ',' expression ',' expression ')' statement
    ;

jump_statement:
      T_RETURN optional_expression ';'
    | T_BREAK ';'
    | T_CONTINUE ';'
    ;

optional_expression:
      expression
    | /* epsilon */
    ;

/* An expression followed by a semicolon */
expression_statement:
      expression ';' 
      ;

/* --- EXPRESSION HIERARCHY (UNIFIED MODEL) --- */
expression:
      assignment_expression
      ;

assignment_expression:
      logical_or_expression
    | postfix_expression '=' assignment_expression /* Simplified assignment */
    ;

logical_or_expression:
      logical_and_expression
    | logical_or_expression T_OR logical_and_expression
    ;

logical_and_expression:
      equality_expression
    | logical_and_expression T_AND equality_expression
    ;

equality_expression:
      relational_expression
    | equality_expression T_EQUAL_EQUAL relational_expression
    | equality_expression T_NOT_EQUAL relational_expression
    ;

relational_expression:
      additive_expression
    | relational_expression '<' additive_expression
    | relational_expression '>' additive_expression
    | relational_expression T_LESS_EQUAL additive_expression
    | relational_expression T_GREATER_EQUAL additive_expression
    ;

additive_expression:
      multiplicative_expression
    | additive_expression '+' multiplicative_expression
    | additive_expression '-' multiplicative_expression
    ;

multiplicative_expression:
      graph_binary_expression
    | multiplicative_expression '*' graph_binary_expression
    | multiplicative_expression '/' graph_binary_expression
    | multiplicative_expression '%' graph_binary_expression
    ;

/* Custom level for our binary graph operators */
graph_binary_expression:
      power_expression
    | graph_binary_expression T_AMPERSAND power_expression
    | graph_binary_expression T_TENSOR_PRODUCT power_expression
    ;

power_expression:
      unary_expression
    | power_expression T_POWER unary_expression
    ;

unary_expression:
      postfix_expression
    | '-' unary_expression %prec UNARY
    | '!' unary_expression %prec UNARY
    | T_COMPLEMENT unary_expression %prec UNARY /* e.g., ~G */
    | T_CONNECTED_COMP unary_expression %prec UNARY /* e.g., #G */
    ;

postfix_expression:
      primary_expression
    | postfix_expression T_DOT T_IDENTIFIER { printf("DEBUG: Member access with identifier %s\n", $3); }
    | postfix_expression T_DOT T_BUILT_IN { printf("DEBUG: Built-in member access for %s\n", $3); }
    | postfix_expression '(' optional_argument_list ')' { printf("DEBUG: Function call\n"); }
    | postfix_expression T_TRANSPOSE /* e.g., G~T */
    | postfix_expression T_TRANSITIVE_CLOSURE /* e.g., G~* */
    | postfix_expression T_METRIC_CLOSURE /* e.g., G~@ */
    ;

primary_expression:
      T_IDENTIFIER
    | T_INTEGER_LITERAL
    | T_FLOAT_LITERAL
    | T_STRING_LITERAL
    | T_CHAR_LITERAL
    | T_TRUE
    | T_FALSE
    | T_NEW type_specifier /* e.g., new graph */
    | '(' expression ')'
    ;

optional_argument_list:
      argument_expression_list
    | /* epsilon */
    ;

argument_expression_list:
      expression
    | argument_expression_list ',' expression
    ;

%%
/* SECTION 3: USER CODE */

/* EDITED: The problematic #include "lex.yy.c" has been REMOVED.
   The linker will handle connecting the two files. */

/* The error reporting function. */
void yyerror(const char *s) {
    fprintf(stderr, "Parse Error at line %d: %s\n", yylineno, s);
}

/* The main function to run the compiler front-end. */
int main(int argc, char **argv) {
  /* This global is enabled by the YYDEBUG define. */
    //extern int yydebug;
    /* STEP 2: This turns on the trace messages at runtime. */
    //yydebug = 1;
    if (argc > 1) {
        FILE *inputFile = fopen(argv[1], "r");
        if (!inputFile) {
            perror(argv[1]);
            return 1;
        }
        yyin = inputFile; /* yyin is the global input file pointer for Flex */
    }
    
    /* yyparse() is the function generated by Yacc/Bison that starts parsing.
       It will call yylex() automatically. */
    if (yyparse() == 0) {
        printf("Parsing complete. The input is valid.\n");
    } else {
        printf("Parsing failed.\n");
    }
    
    return 0;
}
