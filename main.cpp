#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdio>

using namespace std;

#define GNUPLOT_NAME "D:/gnuplot/bin/gnuplot -persist"

/*
 * Defining all classes and their friendly functions
 */

template<typename T>
class Point;

template<typename T>
class Matrix;

template<typename T>
class SquareMatrix;

template<typename T>
class IdentityMatrix;

template<typename T>
class PermutationMatrix;

template<typename T>
class EliminationMatrix;

template<typename T>
class ColumnVector;

template<typename T>
istream &operator>>(istream &in, const Matrix<T> &new_matrix);

template<typename T>
ostream &operator<<(ostream &out, const Matrix<T> &out_matrix);

template<>
ostream &operator<<(ostream &out, const Matrix<double> &out_matrix);

template<typename T>
Matrix<T> operator+(Matrix<T> first_matrix, const Matrix<T> &second_matrix);

template<typename T>
Matrix<T> operator-(Matrix<T> first_matrix, const Matrix<T> &second_matrix);

template<typename T>
Matrix<T> operator*(const Matrix<T> &first_matrix, const Matrix<T> &second_matrix);

template<typename T>
Matrix<T> operator*(const SquareMatrix<T> &first_matrix, const Matrix<T> &second_matrix);

template<typename T>
Matrix<T> operator*(const Matrix<T> &first_matrix, const SquareMatrix<T> &second_matrix);

template<typename T>
istream &operator>>(istream &in, SquareMatrix<T> &new_matrix);

template<typename T>
SquareMatrix<T> operator+(SquareMatrix<T> first_matrix, const SquareMatrix<T> &second_matrix);

template<typename T>
SquareMatrix<T> operator-(SquareMatrix<T> first_matrix, const SquareMatrix<T> &second_matrix);

template<typename T>
SquareMatrix<T> operator*(const SquareMatrix<T> &first_matrix, const SquareMatrix<T> &second_matrix);

template<typename T>
istream &operator>>(istream &in, ColumnVector<T> &vector);

template<typename T>
ostream &operator<<(ostream &out, const ColumnVector<T> &vector);

template<>
ostream &operator<<(ostream &out, const ColumnVector<double> &vector);

template<typename T>
ColumnVector<T> operator+(ColumnVector<T> first_vector, const ColumnVector<T> &second_vector);

template<typename T>
ColumnVector<T> operator-(ColumnVector<T> first_vector, const ColumnVector<T> &second_vector);

template<typename T>
ColumnVector<T> operator*(const T &coefficient, ColumnVector<T> vector);

template<typename T>
ColumnVector<T> operator*(ColumnVector<T> first_vector, const ColumnVector<T> &second_vector);

template<typename T>
ColumnVector<T> operator*(const Matrix<T> &matrix, const ColumnVector<T> &vector);

template<typename T>
istream &operator>>(istream &in, Point<T> &vector);

template<typename T>
ostream &operator<<(ostream &out, const Point<T> &vector);

template<typename T>
class Matrix {
public:
    Matrix() {
        m_cols = 0;
        m_rows = 0;
    }

    Matrix(const int &row, const int &column) {
        if (row <= 0 || column <= 0) {
            throw runtime_error("Invalid m_matrix m_size in constructor");
        }

        m_cols = column;
        m_rows = row;

        m_matrix.resize(m_rows, vector<T>(m_cols));
    }

    //Initialization of m_matrix with a given m_matrix from vectors
    explicit Matrix(const vector<vector<T>> &v) {
        m_rows = v.size();
        m_cols = v[0].m_size();

        //Checking the dimensions of a given function
        //If not all m_cols have the same m_size, throw an error
        for (const auto &column: v) {
            if (column.m_size() != m_cols) {
                throw runtime_error("Invalid matrix!");
            }
        }

        m_matrix = v;
    }

    friend istream &operator>><>(istream &in, const Matrix<T> &new_matrix);

    friend ostream &operator<<<>(ostream &out, const Matrix<T> &out_matrix);

    //Defining the special implementation of output for double matrices
    //Used to cope with -0.00 error
    friend ostream &operator<< <double>(ostream &out, const Matrix<T> &out_matrix);

    //Defining two ways of getting a value of vector with [] operator - const and non-const
    vector<T> &operator[](int pos) {
        return m_matrix[pos];
    }

    vector<T> operator[](int pos) const {
        return m_matrix[pos];
    }

    friend Matrix<T> operator+<>(Matrix<T> first_matrix, const Matrix<T> &second_matrix);

    friend Matrix<T> operator-<>(Matrix<T> first_matrix, const Matrix<T> &second_matrix);

    friend Matrix<T> operator*<>(const Matrix<T> &first_matrix, const Matrix<T> &second_matrix);

    friend Matrix<T> operator*<>(const SquareMatrix<T> &first_matrix, const Matrix<T> &second_matrix);

    friend Matrix<T> operator*<>(const Matrix<T> &first_matrix, const SquareMatrix<T> &second_matrix);

    Matrix &operator=(const Matrix &other_matrix) {
        m_cols = other_matrix.m_cols;
        m_rows = other_matrix.m_rows;
        m_matrix = other_matrix.m_matrix;

        return *this;
    }

    void change(const int &row, const int &column, const T &value) {
        if (!is_indexes_valid(row, column)) {
            throw runtime_error("Invalid indexes");
        }

        m_matrix[row][column] = value;
    }

    //This method is virtual, as it might be overloaded by child classes (f.e. square m_matrix)
    virtual T &get(const int &row, const int &column) {
        if (!is_indexes_valid(row, column)) {
            throw runtime_error("Invalid indexes");
        }

        return m_matrix[row][column];
    }

    int cols() const {
        return m_cols;
    }

    int rows() const {
        return m_rows;
    }

    vector<vector<T>> matrix() const {
        return m_matrix;
    }

    Matrix t() const {
        Matrix new_matrix(m_cols, m_rows);

        for (int i = 0; i < m_rows; ++i) {
            for (int k = 0; k < m_cols; ++k) {
                new_matrix.m_matrix[k][i] = m_matrix[i][k];
            }
        }

        return new_matrix;
    }

protected:
    int m_cols, m_rows;
    vector<vector<T>> m_matrix;

    //Constant with possible in double matrices
    constexpr const static double eps = 1e-4;

    //Reading the m_matrix of defined m_size from the input
    //This operation is written outside the operator>> function, as it might be used by child classes
    istream &insert_matrix(istream &in) {
        for (auto &columns: m_matrix) {
            for (auto &element: columns) {
                in >> element;
            }
        }
        return in;
    }

    //Validating the indexes
    bool is_indexes_valid(const int &row, const int &column) const {
        return row >= 0 && row < m_rows && column >= 0 && column < m_cols;
    }
};

template<typename T>
istream &operator>>(istream &in, Matrix<T> &new_matrix) {
    //If the m_size of the m_matrix is not defined, read it from input and validate it
    if (new_matrix.m_rows == 0 || new_matrix.m_cols == 0) {
        in >> new_matrix.m_rows >> new_matrix.m_cols;

        if (new_matrix.m_rows <= 0 || new_matrix.m_cols <= 0) {
            throw runtime_error("Invalid m_matrix m_size in input");
        }
        new_matrix.m_matrix.resize(new_matrix.m_rows, vector<T>(new_matrix.m_cols));
    }

    return new_matrix.insert_matrix(in);
}

template<typename T>
ostream &operator<<(ostream &out, const Matrix<T> &out_matrix) {
//    out << fixed << setprecision(2);
    for (const auto &columns: out_matrix.m_matrix) {
        for (const auto &element: columns) {
            out << element << ' ';
        }
        out << '\n';
    }
    return out;
}

template<>
ostream &operator<<(ostream &out, const Matrix<double> &out_matrix) {
//    out << fixed << setprecision(2);
    for (const auto &columns: out_matrix.m_matrix) {
        for (const auto &element: columns) {
            //If given element is close enough to 0, print it as '0.00' (avoiding '-0.00' interpretation)
            if (abs(element) < Matrix<double>::eps) {
                out << 0.0 << ' ';
            } else {
                out << element << ' ';
            }
        }
        out << '\n';
    }

    return out;
}

template<typename T>
Matrix<T> operator+(Matrix<T> first_matrix, const Matrix<T> &second_matrix) {
    if (first_matrix.m_cols != second_matrix.m_cols ||
        first_matrix.m_rows != second_matrix.m_rows) {
        throw runtime_error("Invalid sizes of m_matrix");
    }

    for (int row = 0; row < first_matrix.m_rows; ++row) {
        for (int column = 0; column < second_matrix.m_cols; ++column) {
            first_matrix.m_matrix[row][column] += second_matrix.m_matrix[row][column];
        }
    }

    return first_matrix;
}

template<typename T>
Matrix<T> operator-(Matrix<T> first_matrix, const Matrix<T> &second_matrix) {
    if (first_matrix.m_cols != second_matrix.m_cols ||
        first_matrix.m_rows != second_matrix.m_rows) {
        throw runtime_error("Invalid sizes of m_matrix");
    }

    for (int row = 0; row < first_matrix.m_cols; ++row) {
        for (int column = 0; column < first_matrix.m_cols; ++column) {
            first_matrix.m_matrix[row][column] -= second_matrix.m_matrix[row][column];
        }
    }

    return first_matrix;
}

template<typename T>
Matrix<T> operator*(const Matrix<T> &first_matrix, const Matrix<T> &second_matrix) {
    if (first_matrix.m_cols != second_matrix.m_rows) {
        throw runtime_error("Invalid sizes of matrices");
    }

    // Creating a new m_matrix, as the multiplication of two matrices in common case will produce
    // a m_matrix with different dimentions
    Matrix new_matrix = Matrix<T>(first_matrix.m_rows, second_matrix.m_cols);

    for (int row = 0; row < first_matrix.m_rows; ++row) {
        for (int column = 0; column < second_matrix.m_cols; ++column) {
            for (int iterator = 0; iterator < first_matrix.m_cols; ++iterator) {
                new_matrix.m_matrix[row][column] +=
                        first_matrix.m_matrix[row][iterator] * second_matrix.m_matrix[iterator][column];
            }
        }
    }

    return new_matrix;
}

template<typename T>
class SquareMatrix : public Matrix<T> {
public:
    SquareMatrix() : Matrix<T>(), m_size(0) {}

    explicit SquareMatrix(const int &size) : Matrix<T>(size, size), m_size(size) {}

    //Initializing a m_matrix with given vector
    explicit SquareMatrix(const vector<vector<T>> &v) : Matrix<T>(v) {
        // Checking if a given vector is a square m_matrix
        // Matrix<T> initialisation guarantees that this is a valid m_matrix NxM, so it is enough to check
        // first row and column
        if (v.size() != v[0].m_size()) {
            throw runtime_error("Given function is not square");
        }
        m_size = v.size();
    }

    //Constructor of SquareMatrix by giving an usual m_matrix
    //Defines downcast Matrix->SquareMatrix
    SquareMatrix(const Matrix<T> &mat) {
        // Check if a given Matrix is a square m_matrix
        if (mat.cols() != mat.rows()) {
            throw runtime_error("Upcast Matrix to_row SquareMatrix error");
        }

        Matrix<T>::m_cols = mat.cols();
        Matrix<T>::m_rows = mat.rows();
        Matrix<T>::m_matrix = mat.matrix();

        m_size = mat.cols();
    }

    // Redefining the input operator, as square m_matrix m_size could be initialized by only one variable
    friend istream &operator>><>(istream &in, SquareMatrix<T> &new_matrix);

    friend SquareMatrix operator+<>(SquareMatrix<T> first_matrix, const SquareMatrix<T> &second_matrix);

    friend SquareMatrix operator-<>(SquareMatrix<T> first_matrix, const SquareMatrix<T> &second_matrix);

    friend SquareMatrix operator*<>(const SquareMatrix<T> &first_matrix, const SquareMatrix<T> &second_matrix);

    SquareMatrix t() {
        return SquareMatrix(Matrix<T>::t());
    }

    SquareMatrix &operator=(const SquareMatrix &other_matrix) {
        Matrix<T>::m_cols = other_matrix.m_cols;
        Matrix<T>::m_rows = other_matrix.m_rows;
        Matrix<T>::m_matrix = other_matrix.m_matrix;

        m_size = other_matrix.m_size;

        return *this;
    }

    int size() const {
        return m_size;
    }

protected:
    int m_size;
};

template<typename T>
istream &operator>>(istream &in, SquareMatrix<T> &new_matrix) {
    if (new_matrix.m_size == 0) {
        in >> new_matrix.m_size;
        if (new_matrix.m_size <= 0) {
            throw runtime_error("Invalid square m_matrix m_size");
        }

        new_matrix.m_cols = new_matrix.m_size;
        new_matrix.m_rows = new_matrix.m_size;
        new_matrix.m_matrix.resize(new_matrix.m_size, vector<T>(new_matrix.m_size));
    }

    return new_matrix.insert_matrix(in);
}

template<typename T>
SquareMatrix<T> operator+(const SquareMatrix<T> &first_matrix, const SquareMatrix<T> &second_matrix) {
    return (SquareMatrix<T>) ((Matrix<T>) first_matrix + (Matrix<T>) second_matrix);
}

template<typename T>
SquareMatrix<T> operator-(const SquareMatrix<T> &first_matrix, const SquareMatrix<T> &second_matrix) {
    return (SquareMatrix<T>) ((Matrix<T>) first_matrix - (Matrix<T>) second_matrix);
}

template<typename T>
SquareMatrix<T> operator*(const SquareMatrix<T> &first_matrix, const SquareMatrix<T> &second_matrix) {
    return SquareMatrix((Matrix<T>) first_matrix * (Matrix<T>) second_matrix);
}
template<typename T>
Matrix<T> operator*(const SquareMatrix<T> &first_matrix, const Matrix<T> &second_matrix) {
    return (Matrix<T>) first_matrix + second_matrix;
}
template<typename T>
Matrix<T> operator*(const Matrix<T> &first_matrix, const SquareMatrix<T> &second_matrix) {
    return first_matrix * (Matrix<T>) second_matrix;
}

template<typename T>
class IdentityMatrix : public SquareMatrix<T> {
public:
    explicit IdentityMatrix(const int &size) : SquareMatrix<T>(size) {
        for (int i = 0; i < size; ++i) {
            for (int k = 0; k < size; ++k) {
                if (i == k) {
                    Matrix<T>::m_matrix[i][k] = T(1);
                } else {
                    Matrix<T>::m_matrix[i][k] = T(0);
                }
            }
        }
    }
};

template<typename T>
class PermutationMatrix : public IdentityMatrix<T> {
public:
    PermutationMatrix(int size, int from, int to) : IdentityMatrix<T>(size), from_row(from), to_row(to) {
        //Checking the validness of changed columns
        if (!Matrix<T>::is_indexes_valid(from_row, to_row)) {
            throw runtime_error("Invalid permutation m_matrix constructor");
        }

        //Changes the position of cells with '1' value <==> changing the columns
        //as all other values in given columns of identity m_matrix are 0
        int tmp = Matrix<T>::m_matrix[from_row][from_row];
        Matrix<T>::m_matrix[from_row][from_row] = Matrix<T>::m_matrix[to_row][from_row];
        Matrix<T>::m_matrix[to_row][from_row] = tmp;

        tmp = Matrix<T>::m_matrix[to_row][to_row];
        Matrix<T>::m_matrix[to_row][to_row] = Matrix<T>::m_matrix[from_row][to_row];
        Matrix<T>::m_matrix[from_row][to_row] = tmp;
    }

private:
    int from_row, to_row;
};

template<typename T>
class EliminationMatrix : public IdentityMatrix<T> {
public:
    EliminationMatrix(vector<vector<T>> eliminated_matrix, int row, int col) :
      IdentityMatrix<T>(eliminated_matrix.size()),
      eliminate_row(row),
      eliminate_col(col) {
        //Checking the validness of changed columns
        if (!Matrix<T>::is_indexes_valid(row, col)) {
            throw runtime_error("Invalid elimination m_matrix constructor");
        }

        //By definition of elimination m_matrix, it is identity m_matrix, where index with element we want to eliminate
        //equals to -(value in this cell in a given m_matrix)/(pivot of this column)
        //Yet, pivot is an element on the diagonal
        //TODO: rewrite this code for cases, when pivot is not diagonal element
        Matrix<T>::m_matrix[eliminate_row][eliminate_col] =
                -eliminated_matrix[eliminate_row][eliminate_col] / eliminated_matrix[eliminate_col][eliminate_col];
    }

    EliminationMatrix(Matrix<T> eliminated_matrix, int row, int col) :
            EliminationMatrix(eliminated_matrix.matrix(), row, col) {}

    EliminationMatrix(SquareMatrix<T> eliminated_matrix, int row, int col) :
            EliminationMatrix(eliminated_matrix.matrix(), row, col) {}

private:
    int eliminate_row, eliminate_col;
};

template<typename T>
class ColumnVector {
public:
    ColumnVector() {
        dims = 0;
    }

    explicit ColumnVector(int dimension) {
        if (dimension < 0) {
            throw runtime_error("Invalid vector m_size");
        }
        dims = dimension;
        coordinates = vector<T>(dims);
    }

    explicit ColumnVector(vector<T> v) : dims(v.size()), coordinates(v) {}

    friend istream &operator>><>(istream &in, ColumnVector<T> &vector);

    friend ostream &operator<<<>(ostream &out, const ColumnVector<T> &vector);
    //Defining the special implementation of output for double matrices
    //Used to cope with -0.00 error
    friend ostream &operator<<<>(ostream &out, const ColumnVector<double> &vector);

    friend ColumnVector<T> operator+<>(ColumnVector<T> first_vector, const ColumnVector<T> &second_vector);

    friend ColumnVector<T> operator-<>(ColumnVector<T> first_vector, const ColumnVector<T> &second_vector);

    friend ColumnVector<T> operator*<>(const T &coefficient, ColumnVector<T> vector);

    friend ColumnVector<T> operator*<>(ColumnVector<T> first_vector, const ColumnVector<T> &second_vector);

    friend ColumnVector<T> operator*<>(const Matrix<T> &matrix, const ColumnVector<T> &vector);

    int size() const {
        return dims;
    }

    T &operator[](int idx) {
        if (!is_index_valid(idx)) {
            throw runtime_error("Invalid operator");
        }

        return coordinates[idx];
    }

    T operator[](int idx) const {
        if (!is_index_valid(idx)) {
            throw runtime_error("Invalid operator");
        }

        return coordinates[idx];
    }

    ColumnVector &operator=(const ColumnVector &other_matrix) {
        coordinates = other_matrix.coordinates;
        dims = other_matrix.dims;
        return *this;
    }

    T dot(const ColumnVector<T> &multiplied_vector) const {
        if (multiplied_vector.size() != dims) {
            throw runtime_error("Different dimension in a dot product of vectors");
        }

        T answer = 0;
        for (int i = 0; i < dims; ++i) {
            answer += coordinates[i] * multiplied_vector[i];
        }
        return answer;
    }

    T abs() const {
        //Absolute value of a vector is equal to
        //square root of given vector with itself
        return sqrt(dot(*this));
    }

protected:
    vector<T> coordinates;
    int dims;
    constexpr static double eps = 1e-4;

    //Checking the validness of given index
    bool is_index_valid(int idx) const {
        return (idx >= 0) && (idx < dims);
    }
};

template<typename T>
istream &operator>>(istream &in, ColumnVector<T> &vector) {
    if (vector.dims == 0) {
        in >> vector.dims;
        if (vector.dims < 0) {
            throw runtime_error("Invalid vector m_size");
        }
        vector.coordinates.resize(vector.dims);
    }

    for (int i = 0; i < vector.dims; ++i) {
        in >> vector[i];
    }

    return in;
}

template<typename T>
ostream &operator<<(ostream &out, const ColumnVector<T> &vector) {
//    out << fixed << setprecision(2);
    for (int i = 0; i < vector.dims; ++i) {
        out << vector[i] << '\n';
    }
    return out;
}

template<>
ostream &operator<<(ostream &out, const ColumnVector<double> &vector) {
//    out << fixed << setprecision(2);
    for (int i = 0; i < vector.dims; ++i) {
        //If given element is close enough to 0, print it as '0.00' (avoiding '-0.00' interpretation)
        if (abs(vector[i]) < ColumnVector<double>::eps) {
            cout << 0.0 << '\n';
        } else {
            out << vector[i] << '\n';
        }
    }
    return out;
}

template<typename T>
ColumnVector<T> operator+(ColumnVector<T> first_vector, const ColumnVector<T> &second_vector) {
    if (first_vector.dims != second_vector.dims) {
        throw runtime_error("Invalid vector's dimentions");
    }
    for (int i = 0; i < first_vector.dims; ++i) {
        first_vector[i] += second_vector[i];
    }

    return first_vector;
}

template<typename T>
ColumnVector<T> operator*(const T &coefficient, ColumnVector<T> vector) {
    for (auto &element: vector.vector) {
        element *= coefficient;
    }
    return vector;
}

template<typename T>
ColumnVector<T> operator-(ColumnVector<T> first_vector, const ColumnVector<T> &second_vector) {
    if (first_vector.dims != second_vector.dims) {
        throw runtime_error("Invalid vector's dimentions");
    }
    for (int i = 0; i < first_vector.dims; ++i) {
        first_vector[i] -= second_vector[i];
    }

    return first_vector;
}

template<typename T>
ColumnVector<T> operator*(ColumnVector<T> first_vector, const ColumnVector<T> &second_vector) {
    if (first_vector.dims != second_vector.dims) {
        throw runtime_error("Invalid vector's dimentions");
    }

    for (int i = 0; i < first_vector.dims; ++i) {
        first_vector[i] *= second_vector[i];
    }

    return first_vector;
}

template<typename T>
ColumnVector<T> operator*(const Matrix<T> &matrix, const ColumnVector<T> &vector) {
    if (matrix.cols() != vector.dims) {
        throw runtime_error("Invalid dimentions of m_matrix and vector");
    }
    ColumnVector<T> answer(matrix.rows());

    answer.coordinates.resize(matrix.rows(), 0);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int k = 0; k < vector.dims; ++k) {
            answer[i] += vector[k] * matrix[i][k];
        }
    }

    return answer;
}

template<typename T>
class Point{
public:
    Point() = default;
    Point (T x, T y): p_x(x), p_y(y){}

    T x() const{
        return p_x;
    }

    T y() const {
        return p_y;
    }

    friend istream &operator>> <>(istream &in, Point<T> &point);
    friend ostream &operator<< <>(ostream &out, const Point<T> &point);

    void set_y(T value) {
        p_y = value;
    }

    void set_x(T value) {
        p_x = value;
    }
private:
    T p_x, p_y;
};

template<typename T>
istream &operator>> (istream &in, Point<T> &point) {
    in >> point.p_x >> point.p_y;
    return in;
}
template<typename T>
ostream &operator<< (ostream &out, const Point<T> &point) {
    out << point.p_x << " " << point.p_y;
    return out;
}

//----------------------------------------------------------------------------
//Functions realisation

//function, which finds the row with maximum element in it (by absolute value), started from the diagonal of m_matrix
int find_max_column_value(const SquareMatrix<double> &matrix, int column) {
    int max_row = column;

    for (int i = column + 1; i < matrix.rows(); ++i) {
        if (abs(matrix[i][column]) > abs(matrix[max_row][column])) {
            max_row = i;
        }
    }

    return max_row;
}

//Concatenates two square matrices
Matrix<double> concatenate(const SquareMatrix<double> &first_matrix, const SquareMatrix<double> &second_matrix) {
    //Required m_matrix should have dimension m_rows X (2*m_cols)
    auto answer = Matrix<double>(first_matrix.size(), first_matrix.size() * 2);

    for (int i = 0; i < answer.rows(); ++i) {
        for (int k = 0; k < first_matrix.size(); ++k) {
            answer[i][k] = first_matrix[i][k];
        }
        for (int k = first_matrix.size(); k < answer.cols(); ++k) {
            answer[i][k] = second_matrix[i][k - answer.rows()];
        }
    }

    return answer;
}

//Calculating an triangular m_matrix
SquareMatrix<double> upper_triangular(SquareMatrix<double> matrix, int &iteration_step, bool silent = true) {
    for (int i = 0; i < matrix.cols() - 1; ++i) {
        //finding the maximum column
        int max_row = find_max_column_value(matrix, i);

        //If yes, apply permutation m_matrix for swapping the m_rows
        if (max_row != i) {
            //Applying permutation m_matrix for swapping the m_rows
            matrix = PermutationMatrix<double>(matrix.size(), i, max_row) * matrix;

            cout << "step #" << iteration_step << ": permutation\n" << matrix;
            iteration_step++;
        }

        // For all lower elements perform an elimination
        for (int k = i + 1; k < matrix.rows(); ++k) {
            // Multiply the m_matrix by corresponding elimination m_matrix
            matrix = EliminationMatrix<double>(matrix, k, i) * matrix;

            cout << "step #" << iteration_step << ": elimination\n" << matrix;
            iteration_step++;
        }
    }

    return matrix;
}

double determinant(const SquareMatrix<double> &matrix) {
    int iteration_step = 1;

    //calculating an elimination m_matrix
    auto up_triangular_matrix = upper_triangular(matrix, iteration_step);

    double ans = 1;

    //Determinant of this m_matrix is equal to multiplication of its diagonal elements
    for (int i = 0; i < matrix.size(); ++i) {
        ans *= up_triangular_matrix[i][i];

    }
    return ans;
}

SquareMatrix<double> inverse(SquareMatrix<double> matrix, bool silent = true) {
    //Create an identity m_matrix which will be modified in a same way as given m_matrix
    SquareMatrix<double> i_matrix = IdentityMatrix<double>(matrix.size());

    int iteration_step = 0;
    if(!silent)
    {
        cout << "step #" << iteration_step << ": Augmented Matrix\n" << concatenate(matrix, i_matrix);
        iteration_step++;
        cout << "Direct way:\n";
    }
    for (int i = 0; i < matrix.cols() - 1; ++i) {
        //finding the maximum column
        int max_row = find_max_column_value(matrix, i);

        //Check if we should swap the current pivot with anything
        if (max_row != i) {
            //Calculating a permutation m_matrix and applying it to both m_matrix and identity m_matrix
            auto permutation = PermutationMatrix<double>(matrix.size(), i, max_row);
            matrix = permutation * matrix;
            i_matrix = permutation * i_matrix;
            if(!silent) {
                cout << "step #" << iteration_step << ": permutation\n" << concatenate(matrix, i_matrix);
                iteration_step++;
            }
        }

        // For all lower elements perform an elimination
        for (int k = i + 1; k < matrix.rows(); ++k) {
            if (matrix[k][i] != 0) {
                // Calculating an elimination m_matrix and applying it to mboth m_matrix and identity m_matrix
                auto elimination = EliminationMatrix<double>(matrix, k, i);
                matrix = elimination * matrix;
                i_matrix = elimination * i_matrix;
                if(!silent) {
                    cout << "step #" << iteration_step << ": elimination\n" << concatenate(matrix, i_matrix);
                    iteration_step++;
                }
            }
        }
    }

    if(!silent) {
        cout << "Way back:\n";
    }
    for (int i = matrix.cols() - 1; i > 0; --i) {
        //Perform an elimination for all element over main diagonal
        //Loop body is the same as in previous loop
        for (int k = i - 1; k >= 0; --k) {
            if (matrix[k][i] != 0) {
                auto elimination = EliminationMatrix<double>(matrix, k, i);
                matrix = elimination * matrix;
                i_matrix = elimination * i_matrix;
                if(!silent) {
                    cout << "step #" << iteration_step << ": elimination\n" << concatenate(matrix, i_matrix);
                    iteration_step++;
                }
            }
        }
    }

    //Scaling m_rows in both m_matrix and identity m_matrix
    for (int i = 0; i < matrix.size(); ++i) {
        double scale = matrix[i][i];

        matrix[i][i] /= scale;
        for (int k = 0; k < matrix.size(); ++k) {
            i_matrix[i][k] /= scale;
        }
    }
    if(!silent) {
        cout << "Diagonal normalization:\n" << concatenate(matrix, i_matrix);
    }
    //Return identity m_matrix as it contains inverse of given m_matrix
    return i_matrix;
}

ColumnVector<double> linear_solver(SquareMatrix<double> matrix, ColumnVector<double> vector) {
    int iteration_step = 0;
    cout << "step #" << iteration_step << ":\n" << matrix << vector;
    iteration_step++;

    //This algo is the same as in previous function, but there is vector instead identity matrkx
    for (int i = 0; i < matrix.cols() - 1; ++i) {
        int max_row = find_max_column_value(matrix, i);

        if (max_row != i) {
            auto permutation = PermutationMatrix<double>(matrix.size(), i, max_row);
            matrix = permutation * matrix;
            vector = permutation * vector;

            cout << "step #" << iteration_step << ": permutation\n" << matrix << vector;
            iteration_step++;
        }
        for (int k = i + 1; k < matrix.rows(); ++k) {
            if (matrix[k][i] != 0) {
                auto elimination = EliminationMatrix<double>(matrix, k, i);
                matrix = elimination * matrix;
                vector = elimination * vector;

                cout << "step #" << iteration_step << ": elimination\n" << matrix << vector;
                iteration_step++;
            }
        }
    }

    for (int i = matrix.cols() - 1; i > 0; --i) {
        for (int k = i - 1; k >= 0; --k) {
            if (matrix[k][i] != 0) {
                auto elim = EliminationMatrix<double>(matrix, k, i);
                matrix = elim * matrix;
                vector = elim * vector;
                cout << "step #" << iteration_step << ": elimination\n" << matrix << vector;
                iteration_step++;
            }
        }
    }

    for (int i = 0; i < matrix.size(); ++i) {
        double scale = matrix[i][i];
        matrix[i][i] /= scale;
        vector[i] /= scale;
    }

    cout << "Diagonal normalization:\n" << matrix << vector;

    return vector;
}

ColumnVector<double> Jacobi(const SquareMatrix<double>& matrix, const ColumnVector<double> &free_coefs, const double& final_error) {
    cout << fixed << setprecision(4);
    double eps = 100;

    bool if_applicable = true;
    for(int i = 0; i < matrix.size(); ++i) {
        double sum = 0;
        for(int k = 0; k < matrix.size(); ++k) {
            if(i != k) {
                sum += matrix[i][k];
            }
        }
        if(sum > matrix[i][i]) {
            if_applicable = false;
            break;
        }
    }

    if(!if_applicable) {
        throw runtime_error("The method is not applicable!");
    }

    SquareMatrix<double> alpha(matrix.size());

    for(int i = 0; i < matrix.size(); ++i) {
        for(int k = 0; k < matrix.size(); ++k) {
            if(i == k) {
                alpha[i][k] = 0.0;
            } else {
                alpha[i][k] = -matrix[i][k] / matrix[i][i];
            }
        }
    }
    ColumnVector<double> beta(free_coefs.size());

    for(int i = 0; i < beta.size(); ++i) {
        beta[i] = free_coefs[i] / matrix[i][i];
    }

    cout << "alpha:\n" << alpha << "beta:\n" << beta;

    int cur_iteration = 0;

    ColumnVector<double> x(beta);
    ColumnVector<double> final_x(alpha*x + beta);
    while(eps > final_error) {
        eps = (final_x-x).abs();
        cout << "x(" << cur_iteration << "):\n" << x << "e: " << eps << '\n';
        cur_iteration++;
        x = alpha*x + beta;
        final_x = alpha*x + beta;
    }
    cout << "x(" << cur_iteration << "):\n" << x;
    return final_x;
}

ColumnVector<double> Seidel(const SquareMatrix<double>& matrix, const ColumnVector<double>& free_coefs, const double& final_error) {
    cout << fixed << setprecision(4);
    double eps = 100;
    SquareMatrix<double> alpha(matrix.size());

    bool if_applicable = true;
    for(int i = 0; i < matrix.size(); ++i) {
        double sum = 0;
        for(int k = 0; k < matrix.size(); ++k) {
            if(i != k) {
                sum += matrix[i][k];
            }
        }
        if(sum > matrix[i][i]) {
            if_applicable = false;
            break;
        }
    }

    if(!if_applicable) {
        throw runtime_error("The method is not applicable!");
    }

    for(int i = 0; i < matrix.size(); ++i) {
        for(int k = 0; k < matrix.size(); ++k) {
            if(i == k) {
                alpha[i][k] = 0.0;
            } else {
                alpha[i][k] = -matrix[i][k] / matrix[i][i];
            }
        }
    }
    ColumnVector<double> beta(free_coefs.size());

    for(int i = 0; i < beta.size(); ++i) {
        beta[i] = free_coefs[i] / matrix[i][i];
    }

    cout << "beta:\n" << beta << "alpha:\n" << alpha;

    SquareMatrix<double> B(matrix.size()), C(matrix.size());
    for(int i = 0; i < alpha.size(); ++i) {
        for(int k = 0; k < alpha.size(); ++k) {
            if(i < k) {
                C[i][k] = alpha[i][k];
                B[i][k] = 0;
            } else if(i > k) {
                B[i][k] = alpha[i][k];
                C[i][k] = 0;
            }
            else {
                B[i][k] = 0;
                C[i][k] = 0;
            }
        }
    }
    cout << "B:\n" << B << "C:\n" << C;

    B = Matrix((IdentityMatrix<double>(matrix.size()))) - B; //TODO: ???
    cout << "I-B:\n" << B;
    B = inverse(B);
    cout << "(I-B)_-1:\n" << B;

    int iteration_step = 0;

    ColumnVector<double> x(beta), final_x(B*(C*x + beta));
    while(eps > final_error) {
        eps = (final_x-x).abs();
        cout << "x(" << iteration_step << "):\n" << x << "e: " << eps << '\n';
        iteration_step++;
        x = B*(C*x + beta);
        final_x = B*(C*x + beta);
    }

    cout << "x(" << iteration_step << "):\n" << x;
    return final_x;
}

ColumnVector<double> least_squares(const ColumnVector<Point<double>>& points, const int& degree, bool silent = true) {
    cout << fixed << setprecision(4);

    ColumnVector<double> t(points.size()), b(points.size());
    for(int i = 0; i < points.size(); ++i) {
        t[i] = points[i].x();
        b[i] = points[i].y();
    }

    Matrix<double> A(points.size(), degree + 1);
    for (int i = 0; i < points.size(); ++i) {
        for (int k = 0; k <= degree; ++k) {
            A[i][k] = pow(t[i], k);
        }
    }
    auto aT = A.t() * A;
    auto aT_inverse = inverse(aT);
    auto aTb = A.t() * b;
    auto answer = aT_inverse * aTb;

    if(!silent) {
        cout << "A:\n" << A << "A_T*A:\n" << aT << "(A_T*A)^-1:\n" << aT_inverse << "A_T*b:\n" << aTb <<
        "x~:\n" << answer;
    }
    return answer;
}

double random_coordinate(double lower_limit, double upper_limit) {
    double f = (double)rand() / RAND_MAX;
    return lower_limit + f * (upper_limit - lower_limit);
}

void plotting() {
    FILE* pipe = popen(GNUPLOT_NAME, "w");

    if(pipe) {
        constexpr int points_amount = 20;
        constexpr int polynominal_degree = 5;
        srand(time(nullptr));

        ColumnVector<Point<double>> points(points_amount);
        cout << "Dataset:\n";
        for(int i = 0; i < points.size(); ++i) {
            points[i].set_x(random_coordinate(1, 10));
            points[i].set_y (random_coordinate(1, 10));
            cout << points[i] << '\n';
        }

        ColumnVector<double> polinomial_coefficients(polynominal_degree+1);
        polinomial_coefficients = least_squares(points, polynominal_degree);
        string polinom = "f(x) = ";
        for(int i = 0; i <= polynominal_degree; ++i) {
            polinom += "+ " + to_string(polinomial_coefficients[i]) + " * x**" + to_string(i);
        }

        fprintf(pipe, "%s\n", polinom.c_str());
        fprintf(pipe, "%s\n", "plot [0:10] [0:10] f(x), '-' u 1:2 t 'points' w p pt 7 ps 1");
        for(int i = 0; i < points_amount; ++i) {
            fprintf(pipe, "%f\t%f\n", points[i].x(), points[i].y());
        }
        fprintf(pipe, "%s\n", "e");
    }
    pclose(pipe);
}

int main() {
    SquareMatrix<double> A;
    ColumnVector<double> B;

    cin >> n;
    cin >> A;
//    double eps;
//    cin >> A >> B >> eps;
//    Seidel(A, B, eps);
}
