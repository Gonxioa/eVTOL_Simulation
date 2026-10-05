import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import savemat
from uavfd_f550.common import permit, split
from uavfd_f550.data import Record, author_imu_average
from uavfd_f550.verification import analytical, demo_body


class AnalyticalTests(unittest.TestCase):
    def test_physical_identities(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertTrue(analytical(Path(temp))["all_passed"])

    def test_lock_and_split(self):
        s = split()
        self.assertEqual(sum(map(len, (s["calibration"], s["development"], s["locked_test"]))), 18)
        for name in s["locked_test"]:
            with self.assertRaises(PermissionError):
                permit(name, "development")
            with self.assertRaises(PermissionError):
                permit(name, "fit")
        with self.assertRaises(PermissionError):
            permit("NO_FAULT4.mat", "fit")

    def test_labels_override_column_order_and_average_intersection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"NO_FAULT1.mat"
            labels = np.array(["AccZ", "TimeUS", "GyrY", "AccX", "GyrZ", "GyrX", "AccY"], dtype=object)[:, None]
            variables = {"IMU_label": labels}
            for i, ts in enumerate(([1, 2, 3], [2, 3, 4], [1, 2, 3])):
                x = np.full((3, 7), float(i))
                x[:, 1] = ts
                variables[f"IMU_{i}"] = x
            savemat(path, variables)
            r = Record(path, ["IMU_0", "IMU_1", "IMU_2"])
            t, average, individual = author_imu_average(r)
            np.testing.assert_allclose(t, np.array([2, 3])*1e-6)
            np.testing.assert_allclose(average, 1.)
            self.assertEqual(individual.shape, (3, 2, 6))

    def test_invalid_physics_rejected(self):
        from dataclasses import replace
        from uavfd_f550.dynamics import first_order_step
        from uavfd_f550.vibration import LocalModes, imbalance_force
        b = demo_body()
        with self.assertRaises(ValueError):
            b.wrench(np.full(6, -1.))
        with self.assertRaises(ValueError):
            b.wrench(np.full(6, np.nan))
        for value in (np.nan, np.inf, -1.):
            with self.assertRaises(ValueError):
                replace(b, mass=value)
            with self.assertRaises(ValueError):
                replace(b, gravity=value)
            with self.assertRaises(ValueError):
                b.wrench(np.ones(6), kT_scale=np.full(6, value))
            with self.assertRaises(ValueError):
                first_order_step(0., 100., value, .1)
        with self.assertRaises(ValueError):
            imbalance_force(np.nan, 100., 0., 1)
        with self.assertRaises(ValueError):
            LocalModes([np.nan], [.05], [[1.]], [[1.], [0.], [0.]])
        from uavfd_f550.dynamics import body_from_config
        from uavfd_f550.common import CONFIG, read_json
        with self.assertRaisesRegex(ValueError, "Unknown physical parameters"):
            body_from_config(read_json(CONFIG / "uavfd_aircraft.json"))

    def test_hdf5_matlab_cell_labels(self):
        import h5py
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"NO_FAULT1.mat"
            with h5py.File(path, "w") as f:
                group = f.create_group("#refs#")
                refs = []
                for i, name in enumerate(["TimeUS", "AccZ"]):
                    ds = group.create_dataset(str(i), data=np.array([ord(c) for c in name], dtype=np.uint16)[:, None])
                    ds.attrs["MATLAB_class"] = np.bytes_("char")
                    refs.append(ds.ref)
                label = f.create_dataset("IMU_label", shape=(1, 2), dtype=h5py.ref_dtype)
                label[0] = refs
                label.attrs["MATLAB_class"] = np.bytes_("cell")
                value = f.create_dataset("IMU_0", data=np.array([[1000000., 2000000.], [-9.8, -9.7]]))
                value.attrs["MATLAB_class"] = np.bytes_("double")
            r = Record(path, ["IMU_0"])
            np.testing.assert_allclose(r.table("IMU_0")["AccZ"], [-9.8, -9.7])

    def test_duplicate_requested_mat_variable_rejected(self):
        import io
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"NO_FAULT1.mat"
            savemat(path, {"IMU_0": np.ones((2, 2))})
            extra = io.BytesIO()
            savemat(extra, {"IMU_0": np.zeros((2, 2))})
            with path.open("ab") as f:
                f.write(extra.getvalue()[128:])
            with self.assertRaisesRegex(ValueError, "Ambiguous duplicate"):
                Record(path, ["IMU_0"])


if __name__ == "__main__":
    unittest.main()
