import { StyleSheet } from 'react-native';
import { AppColors } from '../../utils/AppCollors';
import { height, width } from '../../methord/Dimentions';

const styles = StyleSheet.create({
  container: {
    width: width(100),
    height: height(100),
    justifyContent: 'center',
    alignItems: 'center',
    // paddingHorizontal: width(5),
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  inputContainer: {
    width: width(90),
    marginBottom: 10,
  },
  label: {
    marginBottom: 10,
    fontWeight: 'bold',
  },
  input: {
    borderWidth: 1,
    borderColor: '#3C3C3C',
    borderRadius: 5,
    padding: 10,
    width: width(90),
  },
  registerButton: {
    backgroundColor: AppColors.primary,
    width: width(90),
    borderRadius: 8,
    padding: 15,
    alignItems: 'center',
    marginTop: 30,
    elevation: 10,
  },
  registerButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  forgotPassword: {
    width: width(90),

    justifyContent: 'flex-end',
    flexDirection: "row",
    marginTop: height(2),

  },
  error: {
    width: width(90),

    justifyContent: 'flex-end',
    flexDirection: "row",

  },
  forgotPasswordText: {
    color: 'black',
  },
  loginText: {
    color: AppColors.primary,
    //   fontSize: 18,
    //   fontWeight: 'bold',
  }
});

export default styles;
