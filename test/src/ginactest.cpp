#include <iostream>
#include <ginac/ginac.h>

// Forward declaration of our custom function.
class Int;
DECLARE_FUNCTION_2P(Int);
class Int : public GiNaC::function {
public:
    // Constructor: use the serial from IntFunction_SERIAL (declared by the macro)
    explicit Int(const GiNaC::ex& integrand, const GiNaC::ex& variable)
        : GiNaC::function(Int_SERIAL::serial, GiNaC::lst{integrand, variable}) {}

    // Differentiation rule: d/dt Int(f, t) = f.
    GiNaC::ex derivative(const GiNaC::symbol& s) const override {
        const GiNaC::ex& var = this->op(1);
        if (s.is_equal(GiNaC::ex_to<GiNaC::symbol>(var))) {
            return this->op(0);  // Fundamental theorem of calculus.
        } else {
            // Apply chain rule for other variables.
            return Int(this->op(0).diff(s), var) * var.diff(s);
        }
    }

    // Pretty-printing.
    void print(const GiNaC::print_context& c, unsigned level) const override {
        c.s << "Int(";
        this->op(0).print(c);
        c.s << ", ";
        this->op(1).print(c);
        c.s << ")";
    }
};
REGISTER_FUNCTION(Int, dummy());

int main(int argc, char ** argv)
{
    // GiNaC::symbol x("x"), y("y"), t("t");
    // GiNaC::ex expr = x + Int(y, t) == 2*x;
    // std::cout << "expr: " << expr << std::endl;
    // auto s = GiNaC::lsolve(expr, y);
    // std::cout << "s: x == " << s << std::endl;

    GiNaC::symbol x("x"), y("y"), t("t"), U("U");
    // Original equation with the Int function
    GiNaC::ex eq_orig = x + Int(y, t) == 2*x*y;
    std::cout << eq_orig << std::endl;
    // Substitute U for Int(y, t)
    auto eq_subst = eq_orig.subs(Int(y, t) == U);
    std::cout << eq_subst << std::endl;
    // Now lsolve should be able to solve eq_subst for U:
    auto solU = GiNaC::lsolve(eq_subst, y);
    std::cout << "y == " << solU << std::endl;
    eq_subst = solU.subs(U == Int(y, t));
    std::cout << "y == " << eq_subst << std::endl;
    std::cout << "dydt == " << GiNaC::diff(eq_subst, t) << std::endl;
    return 0;
}