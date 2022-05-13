import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy.printing.pycode import NumPyPrinter
from sympy.utilities.lambdify import _EvaluatorPrinter
from sympy.codegen.ast import AssignmentBase, Symbol, Element, Variable
from sympy.core.function import AppliedUndef
from sympy.core.compatibility import iterable, PY3, string_types
import keyword
import re

t = dynamicsymbols._t

def staticfy_expressions(expr):
    """
    Converts dynamic symbols (i.e., symbols which are implied functions of time)
    to static symbols in expr. Many codegen operations need dynamic symbols
    converted to static.
    """
    dyn_syms = list(sp.physics.mechanics.functions.find_dynamicsymbols(expr))
    dynamic_to_static = {}
    for sym in dyn_syms:
        if (len(sym.args) == 1) and sym.args[0] == t:
            dynamic_to_static[sym] = sp.symbols(sym.__class__.__name__)
    
    if len(dynamic_to_static) == 0:
        return expr
    
    new_expr = expr.subs(dynamic_to_static)
    
    if (sp.Function in expr.__class__.__mro__) and hasattr(expr, 'shape'):
        new_expr.shape = expr.shape
    
    return new_expr

def process_funcs_to_print(funcs_to_print):
    """
    Prepares a set of sympy expressions to be printed with CSE. The `dict`\s
    are modified in-place.

    funcs_to_print is an iterable of dicts with keys:
        'extra_assignments', a dict of exta assignments before CSE
        'input_args', an Array of the input arguments
        'sym_expr', the symbolic expression to print with CSE
    """

    code_print_params = []
    for funcs_to_print_dict in funcs_to_print:
        extras_dict = {} #funcs_to_print_dict['extra_assignments'].copy()
        for key,val in funcs_to_print_dict['extra_assignments'].items():
            #print(key, val)
            static_key = staticfy_expressions(key)
            static_val = staticfy_expressions(val)
            extras_dict[static_key] = static_val
        funcs_to_print_dict['extra_assignments'] = extras_dict
        if funcs_to_print_dict['input_args']:
            funcs_to_print_dict['input_args'] = staticfy_expressions(funcs_to_print_dict['input_args'])
        funcs_to_print_dict['sym_expr'] = staticfy_expressions(funcs_to_print_dict['sym_expr'])

        tot_size = 1
        if hasattr(funcs_to_print_dict['sym_expr'], 'shape'):
            raveled_shape = funcs_to_print_dict['sym_expr'].shape
            for dim in raveled_shape:
                tot_size = tot_size*dim
            funcs_to_print_dict['sym_expr'] = funcs_to_print_dict['sym_expr'].reshape(tot_size).tolist()

        cse_out = sp.cse(funcs_to_print_dict['sym_expr'], order='none')
        cse_subs = {k:v for k,v in cse_out[0]}
        funcs_to_print_dict['extra_assignments'].update(cse_subs)

        if tot_size > 1:
            funcs_to_print_dict['sym_expr'] = sp.Array(cse_out[1]).reshape(*raveled_shape)
        else:
            funcs_to_print_dict['sym_expr'] = cse_out[1][0]

        code_print_params.append( (funcs_to_print_dict['num_name'], funcs_to_print_dict['input_args'], funcs_to_print_dict['sym_expr'], funcs_to_print_dict['extra_assignments']) )
    return code_print_params

class Assignment(AssignmentBase):
    """
    Represents variable assignment for code generation, but can also use Array
    on the LHS and non-defensively allows a function on RHS with indexed LHS.

    Parameters
    ==========

    lhs : Expr
        Sympy object representing the lhs of the expression. These should be
        singular objects, such as one would use in writing code. Notable types
        include Symbol, MatrixSymbol, MatrixElement, and Indexed. Types that
        subclass these types are also supported.

    rhs : Expr
        Sympy object representing the rhs of the expression. This can be any
        type, provided its shape corresponds to that of the lhs. For example,
        a Matrix type can be assigned to MatrixSymbol, but not to Symbol, as
        the dimensions will not align.
    """

    @classmethod
    def _check_args(cls, lhs, rhs):
        """ Check arguments to __new__ and raise exception if any problems found.

        Derived classes may wish to override this.
        """
        from sympy.matrices.expressions.matexpr import (
            MatrixElement, MatrixSymbol)
        from sympy.tensor.indexed import Indexed

        # Tuple of things that can be on the lhs of an assignment
        assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Element, Variable, sp.Array)
        if not isinstance(lhs, assignable):
            raise TypeError("Cannot assign to lhs of type %s." % type(lhs))

        # Indexed types implement shape, but don't define it until later. This
        # causes issues in assignment validation. For now, matrices are defined
        # as anything with a shape that is not an Indexed
        lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs, Indexed)
        rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs, Indexed)

        # If lhs and rhs have same structure, then this assignment is ok
        if lhs_is_mat and not isinstance(rhs, AppliedUndef):
            if not rhs_is_mat:
                raise ValueError("Cannot assign a scalar to a matrix.")
            elif lhs.shape != rhs.shape:
                raise ValueError("Dimensions of lhs and rhs don't align.")
        elif rhs_is_mat and not lhs_is_mat:
            raise ValueError("Cannot assign a matrix to a scalar.")
    
    op = ':='

class ArrayNumPyPrinter(NumPyPrinter):
    def _print_Max(self, expr):
        return '{0}({1})'.format(self._module_format('numpy.maximum'), ','.join(self._print(i) for i in expr.args))

    def _print_Piecewise(self, expr):
        """Piecewise function printer

        from sympy version:
        # If [default_value, True] is a (expr, cond) sequence in a Piecewise object
        #     it will behave the same as passing the 'default' kwarg to select()
        #     *as long as* it is the last element in expr.args.
        # If this is not the case, it may be triggered prematurely.

        which is just not implemented at all...
        """
        if expr.args[-1].cond == True:
            expr_args = expr.args[:-1]
            default_val = expr.args[-1].expr
        else:
            expr_args = expr.args
            default_val = sp.NaN
        exprs = '[{0}]'.format(','.join(self._print(arg.expr) for arg in expr_args))
        conds = '[{0}]'.format(','.join(self._print(arg.cond) for arg in expr_args))
        return '{0}({1}, {2}, default={3})'.format(
            self._module_format('numpy.select'), conds, exprs,
            self._print(default_val))
        
    def _print_Heaviside(self, expr):
        return '(({0}>0.0)+0.5*({0}==0.0))'.format(','.join(self._print(i) for i in expr.args))

    def _print_ImmutableDenseNDimArray(self, expr):
        # return str(expr)
        return "numpy.array(%s)" % (self._print(expr.tolist()),)

    def _print_Assignment(self, expr):
        if isinstance(expr.lhs, sp.Array):
            lhs = expr.lhs
            rhs = expr.rhs

            if isinstance(expr.rhs, sp.Piecewise):
                # Here we modify Piecewise so each expression is now
                # an Assignment, and then continue on the print.
                expressions = []
                conditions = []
                for (e, c) in rhs.args:
                    expressions.append(Assignment(lhs, e))
                    conditions.append(c)
                temp = sp.Piecewise(*zip(expressions, conditions))
                return self._print(temp)

            lhs_code = self._print(sp.flatten(lhs.tolist()))
            # TODO: I think this is where I could make it so that the assignment gets pretty robust to multidimensional inputs
            # when I re-wrote the static trimmability function, I just had to apply .T instead of .ravel, that may be sufficient generally
            # since I already flatten the sympy expression that is being generated.... not sure though.
            rhs_code = self._print(rhs) + '.ravel()'
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))
            
            # I would prefer to have something like this, but isn't clear how
            temp = sp.Dummy()
            lines = [self._print(Assignment(temp, rhs))]
            temp = sp.Indexed(temp.name, lhs.shape)
            for idx, el in enumerate(lhs):
                lines.append('{}[{}]'.format(self._print(temp), 0))
        else:
            if expr.rhs.func == sp.Piecewise:
                # The sympy/printing/codeprinter.py CodePrinter._print_Assignment explicitly
                # folds the assignments into each case, which is never over-written for the NumPyPrinter
                lhs_code = self._print(expr.lhs)
                rhs_code = self._print(expr.rhs)
                return self._get_statement("%s = %s" % (lhs_code, rhs_code))
            return super()._print_Assignment(expr)

class ModuleNumPyPrinter(ArrayNumPyPrinter):
    def __init__(self, for_class=False, **kwargs):
        new_modules = kwargs.pop('function_modules', {})
        new_func_names = kwargs.pop('function_names', {})
        new_constant_modules = kwargs.pop('constant_modules', {})
        new_constant_names = kwargs.pop('constant_names', {})

        super().__init__(**kwargs)

        self.for_class = for_class # --> prepend self to args, assume unknown functions are in self
        # TODO: also assume constants are in self, but need to distinguish from inputs and intermediate computations

        self.known_func_modules = {None: 'numpy'}
        self.known_func_modules.update(new_modules)

        self.known_func_names = new_func_names

        self.known_constant_modules = new_constant_modules
        self.known_constant_names = new_constant_names
        self.functions_not_supported = set()
        self.constants_not_supported = set()

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))
        
    def _print_Symbol(self, expr):
        super_ret = super()._print_Symbol(expr)
        if expr in self.known_constant_modules:
            module = self.known_constant_modules[expr]
            if expr in self.known_constant_names:
                const_name = self.known_constant_names[expr]
            else:
                const_name = super_ret
            if module != '':
                return self._module_format('.'.join([module,const_name]))
            else:
                return const_name
        return super_ret
 
    def _print_Function(self, expr):
        func = expr.func
        if (len(expr.args) == 1) and (expr.args[0] == dynamicsymbols._t):
            return func.__name__
        elif func not in self.known_functions:
            if func in self.known_func_modules:
                module = self.known_func_modules[func]
            elif self.for_class:
                module = 'self'
            else:
                module = ''
            if func in self.known_func_names:
                func_name = self.known_func_names[func]
            else:
                func_name = func.__name__
            if module != '':
                func_text = self._module_format('.'.join([module,func_name]))
            else:
                func_text = func_name
            args = ','.join(self._print(i) for i in expr.args)
            return "{}({})".format(func_text, args)
        else:
            return super()._print_Function(expr)

    def _print_Assignment(self, expr):
        if hasattr(expr.lhs, 'shape'):
            for lhs_element in expr.lhs:
                self.known_constant_modules[lhs_element] = ''
        else:
            self.known_constant_modules[expr.lhs] = ''
        return super()._print_Assignment(expr)

    def __delattr__(self, attr):
        if attr == '_not_supported':
            self.functions_not_supported = self.functions_not_supported.union(self._not_supported)
        return super().__delattr__(attr)
        

class ModulePrinter(_EvaluatorPrinter):
    _safe_star_args = re.compile(r'^\*{0,2}')

    if PY3:
        @classmethod
        def _is_safe_ident(cls, ident):
            return isinstance(ident, string_types) and cls._safe_star_args.sub('',ident).isidentifier() \
                    and not keyword.iskeyword(ident)
    else:
        @classmethod
        def _is_safe_ident(cls, ident):
            return isinstance(ident, string_types) and cls._safe_ident_re.match(cls._safe_star_args.sub('',ident)) \
                and not (keyword.iskeyword(ident) or ident == 'None')

    def codeprint(self, func_arg_expr_s, extra_pre_lines='', extra_post_lines=''):
        printer = self._exprrepr.__self__
        lines = []
        for module in set(printer.known_func_modules.values()) | set(printer.known_constant_modules.values()):
            if module != '' and not (printer.for_class and module == 'self'):
                lines.append("import {}".format(module))
        if extra_pre_lines != '':
            lines.append(extra_pre_lines)
        lines.append("\n")
        for func_arg_expr in func_arg_expr_s:
            if len(func_arg_expr)==4:
                # do dict assignment?? Maybe nothing extra is needed here, if the printer handles it on assignment
                pass
            lines.append(self.doprint(*func_arg_expr))
            func = sp.Function(func_arg_expr[0])
            if func not in printer.known_func_modules:
                printer.known_func_modules[func] = ''
            
            for func in func_arg_expr[2].atoms(AppliedUndef):
                if func.func not in printer.known_func_modules:
                    if (len(func.args) == 1) and (func.args[0] == dynamicsymbols._t):
                        pass
                    else:
                        raise ValueError("Unknown function {} in ModulePrinter".format(func.__class__.__name__))
            for const in func_arg_expr[2].atoms(sp.Symbol):
                if const not in printer.known_constant_modules and const not in func_arg_expr[1]:
                    if const == dynamicsymbols._t:
                        pass
                    else:
                        raise ValueError("Unknown symbol {} in ModulePrinter".format(str(const)))

        if extra_post_lines != '':
            lines.append(extra_post_lines)

        if len(printer.functions_not_supported):
            raise ValueError("Unknown functions in ModulePrinter", printer.functions_not_supported)
        return "\n".join(lines)
        
        
    
    # copied and modified from _EvaluatorPrinter.doprint
    def doprint(self, funcname, args, expr, extra_assignments=None):
        """
        Returns the function definition code as a string.
        
        extra_assignments is a dict of extra assignment definitions or None
        """
        
        from sympy import Dummy
        
        if extra_assignments is None:
            extra_assignments = {}

        funcbody = []

        if not iterable(args):
            args = [args]

        argstrs, expr = self._preprocess(args, expr)

        # Generate argument unpacking and final argument list
        funcargs = []
        unpackings = []

        for argstr in argstrs:
            if iterable(argstr):
                funcargs.append(self._argrepr(Dummy()))
                unpackings.extend(self._print_unpacking(argstr, funcargs[-1]))
            else:
                funcargs.append(argstr)

        if self._exprrepr.__self__.for_class:
            funcargs.insert(0, 'self')

        funcsig = 'def {}({}):'.format(funcname, ', '.join(funcargs))

        # Wrap input arguments before unpacking
        funcbody.extend(self._print_funcargwrapping(funcargs))

        funcbody.extend(unpackings)
        
        # subs_dict = {}
        for lhs, rhs in extra_assignments.items():
            # subbed_rhs = rhs.subs(subs_dict)
            funcbody.append(self._exprrepr(Assignment(lhs, rhs)))
            # subs_dict[lhs] = subbed_rhs

        funcbody.append('return ({})'.format(self._exprrepr(expr)))

        funclines = [funcsig]
        funclines.extend('    ' + line for line in funcbody)

        return '\n'.join(funclines) + '\n'
