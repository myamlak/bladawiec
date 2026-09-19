#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <toml++/toml.hpp>

#ifdef QcxHasHdf5
#include <highfive/H5File.hpp>
#endif

#include <cstddef>
#include <filesystem>
#include <vector>

TEST(JsonSmokeTest, RoundTripsAChemistryShapedDocument) {
    const nlohmann::json doc =
        nlohmann::json::parse(R"({"molecule": {"charge": 0, "symbols": ["H", "H"]}})");

    EXPECT_EQ(doc["molecule"]["charge"].get<int>(), 0);
    EXPECT_EQ(doc["molecule"]["symbols"].size(), 2u);

    const auto reparsed = nlohmann::json::parse(doc.dump());
    EXPECT_EQ(reparsed["molecule"]["symbols"][1].get<std::string>(), "H");
}

TEST(TomlPlusPlusSmokeTest, ParsesAnInputSchemaShapedDocument) {
    const toml::table doc = toml::parse(
        R"(# qcx input-schema-shaped sample (the io module lands in Phase 1)
accuracy = "normal"

[geometry]
file = "h2o.xyz"
)");

    EXPECT_EQ(doc["accuracy"].value<std::string>(), "normal");
    EXPECT_EQ(doc["geometry"]["file"].value<std::string>(), "h2o.xyz");
}

#ifdef QcxHasHdf5
TEST(HighFiveSmokeTest, RoundTripsAVectorThroughHdf5) {
    const auto tmpPath = std::filesystem::temp_directory_path() / "qcx_highfive_smoke.h5";
    const std::vector<double> data = {1.5, -2.25, 3.125};

    {
        HighFive::File file(tmpPath.string(),
                            HighFive::File::ReadWrite | HighFive::File::Create |
                                HighFive::File::Truncate);
        file.createDataSet("smoke", data);
    }

    std::vector<double> reloaded;

    {
        // Scoped so the file is closed before remove(): Windows locks the
        // file for as long as HDF5 holds it open.
        HighFive::File file(tmpPath.string(), HighFive::File::ReadOnly);
        file.getDataSet("smoke").read(reloaded);
    }

    ASSERT_EQ(reloaded.size(), data.size());

    for (std::size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(reloaded[i], data[i]);
    }

    std::filesystem::remove(tmpPath);
}
#endif
