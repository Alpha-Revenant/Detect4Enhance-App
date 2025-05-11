import { Dimensions } from "react-native"

const width = (number) => {
    let fullWidth = Dimensions.get('window').width
    if (number <= 0) return number

    if (number > 0) return (number / 100) * fullWidth

    if (number > 100) return fullWidth
}
const height = (number) => {
    let fullWidth = Dimensions.get('window').height
    if (number <= 0) return number

    if (number > 0) return (number / 100) * fullWidth

    if (number > 100) return fullWidth
}

export { width, height }