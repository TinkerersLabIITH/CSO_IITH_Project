/* SECTION 1: DEFINITIONS */
%{
#include <stdio.h>
#include <string.h> // Required for strdup()

/* This is the crucial link. It includes the token definitions and the YYSTYPE
   struct as defined by the parser (yacc/bison). */
#include "y.tab.h"

/* This is a forward declaration. The parser will define this global variable.
   We just need to tell the lexer that it exists. */
extern YYSTYPE yylval;

/* Global variable to manually track the column number. */
int column = 1;

/* Helper macro for tokens that have a string value. */
#define STRING_TOKEN_ACTION \
    yylval.stringValue = strdup(yytext); \
    column += yyleng;

%}


DIGIT       [0-9]
LETTER      [a-zA-Z_]
IDENTIFIER  {LETTER}({LETTER}|{DIGIT})*
INTEGER     {DIGIT}+
FLOAT       ({DIGIT}+\.{DIGIT}*|{DIGIT}*\.{DIGIT}+)([eE][-+]?{DIGIT}+)?
STRING      \"(\\.|[^"\\])*\"
CHAR	    \'([^\\\']|\\.)\'

%%


"//".* { /* Ignore single-line comments. */ }
\/\*([^*]|\*+[^/])*\*+\/ {
    for(char *p=yytext;*p;p++)
    {
        if(*p=='\n') yylineno++;
    }
}



[ \t]+        { column += yyleng; }
\n            { yylineno++; column = 1; }



"if"            { return T_IF; }
"elif"          { return T_ELIF; }
"else"          { return T_ELSE; }
"for"           { return T_FOR; }
"in"            { return T_IN; }
"range"         { return T_RANGE; }
"while"         { return T_WHILE; }
"funk"          { return T_FUNK; }
"return"        { return T_RETURN; }
"where"         { return T_WHERE; }
"int"           { return T_INT; }
"float"         { return T_FLOAT; }
"char"          { return T_CHAR; }
"string"        { return T_STRING; }
"double"        { return T_DOUBLE; }
"bool"          { return T_BOOL; }
"true"          { return T_TRUE; }
"false"         { return T_FALSE; }
"break"         { return T_BREAK; }
"continue"      { return T_CONTINUE; }
"vector"        { return T_VECTOR; }
"struct"        { return T_STRUCT; }
"vertex_set"    { return T_VERTEX_SET; }
"edge_set"      { return T_EDGE_SET; }
"graph"         { return T_GRAPH; }
"const"         { return T_CONST; }
"var"           { return T_VAR; }
"new"           { return T_NEW; }


"add_edge"|"add_vertex"|"atp"|"bfs"|"blfr"|"bpm"|"bridges"|"cg"|"dij"|"diam"|"epen"|"export"|"get_degree"|"get_edges"|"get_indegree"|"get_inneighbors"|"get_neighbors"|"get_outdegree"|"get_outneighbors"|"get_vertices"|"is_cycle"|"is_edge"|"load"|"mst"|"print"|"remove_edge"|"remove_vertex"|"size"|"sssp"|"topo"|"transpose"|"update_edge"|"vc" {
    printf("builtin %s \n",yytext);STRING_TOKEN_ACTION; return T_BUILT_IN;
}


{FLOAT}         { yylval.floatValue = atof(yytext); column += yyleng; return T_FLOAT_LITERAL; }
{INTEGER}       { yylval.intValue = atoi(yytext); column += yyleng; return T_INTEGER_LITERAL; }
{STRING}        { printf("string %s \n",yytext);STRING_TOKEN_ACTION; return T_STRING_LITERAL; }
{IDENTIFIER}    { printf(" identifier: %s \n",yytext);STRING_TOKEN_ACTION; return T_IDENTIFIER; }
{CHAR}          { STRING_TOKEN_ACTION; return T_CHAR_LITERAL; }

">="            { return T_GREATER_EQUAL; }
"<="            { return T_LESS_EQUAL; }
"=="            { return T_EQUAL_EQUAL; }
"!="            { return T_NOT_EQUAL; }
"&&"            { return T_AND; }
"||"            { return T_OR; }
"~@"            { return T_METRIC_CLOSURE; }
"~*"            { return T_TRANSITIVE_CLOSURE; }
"~T"            { return T_TRANSPOSE; }
"**"            { return T_POWER; }
"~"             { return T_COMPLEMENT; }
"#"             { return T_CONNECTED_COMP; }
"&"             { return T_AMPERSAND; }
"@"             { return T_TENSOR_PRODUCT; }
"."             { return T_DOT; }
[(){},:;=<>+*/-] { return yytext[0]; }


.               { printf("Lexical Error: Unknown character '%s' at line %d, column %d\n", yytext, yylineno, column); return T_UNKNOWN; }

%%
/* SECTION 3: USER CODE */

/* The parser provides main() and this is all yywrap needs to do. */
int yywrap() {
    return 1;
}
