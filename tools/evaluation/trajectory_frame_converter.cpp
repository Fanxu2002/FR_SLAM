#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

struct Options
{
    std::string input;
    std::string output;

    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;

    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 1.0;

    bool have_tx = false;
    bool have_ty = false;
    bool have_tz = false;
    bool have_qx = false;
    bool have_qy = false;
    bool have_qz = false;
    bool have_qw = false;
};

void PrintUsage(const char* program)
{
    std::cout
        << "Usage:\n"
        << program
        << " --input input.csv"
        << " --output output.csv"
        << " --tx TX --ty TY --tz TZ"
        << " --qx QX --qy QY --qz QZ --qw QW\n\n"
        << "Pose convention:\n"
        << "  T_world_output = T_world_input * T_input_output\n\n"
        << "CSV:\n"
        << "  timestamp,x,y,z,qx,qy,qz,qw\n"
        << "or\n"
        << "  time,x,y,z,qx,qy,qz,qw\n";
}

Options ParseOptions(int argc, char** argv)
{
    Options o;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        auto value = [&]() -> std::string
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(
                    "Missing value after " + arg);
            }

            return argv[++i];
        };

        if (arg == "--input")
        {
            o.input = value();
        }
        else if (arg == "--output")
        {
            o.output = value();
        }
        else if (arg == "--tx")
        {
            o.tx = std::stod(value());
            o.have_tx = true;
        }
        else if (arg == "--ty")
        {
            o.ty = std::stod(value());
            o.have_ty = true;
        }
        else if (arg == "--tz")
        {
            o.tz = std::stod(value());
            o.have_tz = true;
        }
        else if (arg == "--qx")
        {
            o.qx = std::stod(value());
            o.have_qx = true;
        }
        else if (arg == "--qy")
        {
            o.qy = std::stod(value());
            o.have_qy = true;
        }
        else if (arg == "--qz")
        {
            o.qz = std::stod(value());
            o.have_qz = true;
        }
        else if (arg == "--qw")
        {
            o.qw = std::stod(value());
            o.have_qw = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        else
        {
            throw std::runtime_error(
                "Unknown argument: " + arg);
        }
    }

    if (o.input.empty() || o.output.empty())
    {
        throw std::runtime_error(
            "--input and --output are required");
    }

    if (!o.have_tx ||
        !o.have_ty ||
        !o.have_tz ||
        !o.have_qx ||
        !o.have_qy ||
        !o.have_qz ||
        !o.have_qw)
    {
        throw std::runtime_error(
            "All transform values are required");
    }

    return o;
}

std::vector<std::string> Split(
    const std::string& line)
{
    std::vector<std::string> result;
    std::stringstream stream(line);
    std::string field;

    while (std::getline(stream, field, ','))
    {
        result.push_back(field);
    }

    return result;
}

std::size_t Column(
    const std::unordered_map<std::string, std::size_t>& columns,
    const std::string& name)
{
    const auto iterator = columns.find(name);

    if (iterator == columns.end())
    {
        throw std::runtime_error(
            "Missing CSV column: " + name);
    }

    return iterator->second;
}

double Number(
    const std::vector<std::string>& fields,
    const std::size_t index,
    const std::size_t line_number)
{
    if (index >= fields.size())
    {
        throw std::runtime_error(
            "Too few CSV fields at line " +
            std::to_string(line_number));
    }

    const double value = std::stod(fields[index]);

    if (!std::isfinite(value))
    {
        throw std::runtime_error(
            "Non-finite value at line " +
            std::to_string(line_number));
    }

    return value;
}

void Convert(const Options& o)
{
    std::ifstream input(o.input);

    if (!input)
    {
        throw std::runtime_error(
            "Cannot open input: " + o.input);
    }

    std::ofstream output(o.output);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot open output: " + o.output);
    }

    std::string header_line;

    if (!std::getline(input, header_line))
    {
        throw std::runtime_error(
            "Input trajectory is empty");
    }

    const std::vector<std::string> header =
        Split(header_line);

    std::unordered_map<std::string, std::size_t> columns;

    for (std::size_t i = 0; i < header.size(); ++i)
    {
        columns[header[i]] = i;
    }

    std::size_t time_index = 0;

    if (columns.count("timestamp") != 0)
    {
        time_index = columns.at("timestamp");
    }
    else if (columns.count("time") != 0)
    {
        time_index = columns.at("time");
    }
    else
    {
        throw std::runtime_error(
            "Missing timestamp/time column");
    }

    const std::size_t x_index = Column(columns, "x");
    const std::size_t y_index = Column(columns, "y");
    const std::size_t z_index = Column(columns, "z");
    const std::size_t qx_index = Column(columns, "qx");
    const std::size_t qy_index = Column(columns, "qy");
    const std::size_t qz_index = Column(columns, "qz");
    const std::size_t qw_index = Column(columns, "qw");

    Eigen::Quaterniond Q_input_output(
        o.qw,
        o.qx,
        o.qy,
        o.qz);

    if (Q_input_output.norm() < 1.0e-12 ||
        !Q_input_output.coeffs().allFinite())
    {
        throw std::runtime_error(
            "Invalid fixed quaternion");
    }

    Q_input_output.normalize();

    const Eigen::Vector3d p_input_output(
        o.tx,
        o.ty,
        o.tz);

    output
        << "timestamp,x,y,z,qx,qy,qz,qw\n";

    output << std::setprecision(16);

    std::size_t count = 0;
    std::size_t line_number = 1;

    std::string line;

    while (std::getline(input, line))
    {
        ++line_number;

        if (line.empty())
        {
            continue;
        }

        const std::vector<std::string> fields =
            Split(line);

        const double timestamp =
            Number(fields, time_index, line_number);

        const Eigen::Vector3d p_world_input(
            Number(fields, x_index, line_number),
            Number(fields, y_index, line_number),
            Number(fields, z_index, line_number));

        Eigen::Quaterniond Q_world_input(
            Number(fields, qw_index, line_number),
            Number(fields, qx_index, line_number),
            Number(fields, qy_index, line_number),
            Number(fields, qz_index, line_number));

        if (Q_world_input.norm() < 1.0e-12 ||
            !Q_world_input.coeffs().allFinite())
        {
            throw std::runtime_error(
                "Invalid input quaternion at line " +
                std::to_string(line_number));
        }

        Q_world_input.normalize();

        // T_world_output =
        //     T_world_input * T_input_output

        const Eigen::Vector3d p_world_output =
            p_world_input +
            Q_world_input * p_input_output;

        const Eigen::Quaterniond Q_world_output =
            (Q_world_input * Q_input_output)
                .normalized();

        output
            << timestamp << ","
            << p_world_output.x() << ","
            << p_world_output.y() << ","
            << p_world_output.z() << ","
            << Q_world_output.x() << ","
            << Q_world_output.y() << ","
            << Q_world_output.z() << ","
            << Q_world_output.w() << "\n";

        ++count;
    }

    if (count == 0)
    {
        throw std::runtime_error(
            "No poses converted");
    }

    std::cout
        << "trajectory_frame_converter SUCCESS\n"
        << "input  = " << o.input << "\n"
        << "output = " << o.output << "\n"
        << "poses  = " << count << "\n"
        << "T_out = T_in * T_input_output\n"
        << "t = ["
        << p_input_output.x() << ", "
        << p_input_output.y() << ", "
        << p_input_output.z() << "]\n"
        << "q_xyzw = ["
        << Q_input_output.x() << ", "
        << Q_input_output.y() << ", "
        << Q_input_output.z() << ", "
        << Q_input_output.w() << "]\n";
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options options =
            ParseOptions(argc, argv);

        Convert(options);

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "trajectory_frame_converter ERROR: "
            << e.what()
            << "\n";

        return EXIT_FAILURE;
    }
}
