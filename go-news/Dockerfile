FROM golang:1.22-alpine AS builder
WORKDIR /build
COPY go.mod ./
RUN go mod tidy
COPY . .
RUN go build -o go-news .

FROM alpine:latest
WORKDIR /app
COPY --from=builder /build/go-news .
EXPOSE 8092
CMD ["./go-news"]
