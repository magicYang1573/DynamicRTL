op_expression = {
    'Cond': '({0}) ? ({1}) : ({2})',         # ternary operator
    'Add': '({0}) + ({1})',                  # addition
    'Sub': '({0}) - ({1})',                  # subtraction
    'Mul': '({0}) * ({1})',                  # multiplication
    'Div': '({0}) / ({1})',                  # division
    'Mod': '({0}) % ({1})',                  # modulo
    'Lt': '({0}) < ({1})',                   # less than
    'Le': '({0}) <= ({1})',                  # less than or equal
    'Gt': '({0}) > ({1})',                   # greater than
    'Ge': '({0}) >= ({1})',                  # greater than or equal
    'Eq': '({0}) == ({1})',                  # equal
    'Neq': '({0}) != ({1})',                 # not equal
    'Eeq': '({0}) === ({1})',
    'Neeq': '({0}) !== ({1})',
    'And': '({0}) && ({1})',                 # logical and
    'Or': '({0}) || ({1})',                  # logical or
    'Not': '!({0})',                         # logical not
    'BitAnd': '({0}) & ({1})',               # bitwise and
    'BitOr': '({0}) | ({1})',                # bitwise or
    'BitXor': '({0}) ^ ({1})',               # bitwise xor
    'BitNXor': '({0}) ~^ ({1})',             # bitwise xnor
    'ShiftLeft': '({0}) << ({1})',           # shift left
    'ShiftRight': '({0}) >> ({1})',          # shift right
    'AshiftLeft': '({0}) <<< ({1})',         # arithmetic shift left
    'AshiftRight': '({0}) >>> ({1})',        # arithmetic shift right
    'LNot': '!({0})',                        # logical not
    'PartSelect': '{0}[{1}:{2}]',            # bit select
    'URxor': '^({0})',                       # reduction xor
    'URand': '&({0})',                       # reduction and
    'URor': '|({0})',                        # reduction or
    'URnand': '~&({0})',                     # reduction nand
    'URnor': '~|({0})',                      # reduction nor
    'Concat': 'TODO',
    'Case': 'TODO'
}
